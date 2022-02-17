//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <clang/Basic/Diagnostic.h>

using namespace Fortran::frontend;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Custom BeginSourceFileAction
//===----------------------------------------------------------------------===//
bool PrescanAction::BeginSourceFileAction() { return RunPrescan(); }

bool PrescanAndParseAction::BeginSourceFileAction() {
  return RunPrescan() && RunParse();
}

bool PrescanAndSemaAction::BeginSourceFileAction() {
  return RunPrescan() && RunParse() && RunSemanticChecks();
}

bool PrescanAndSemaDebugAction::BeginSourceFileAction() {
  // Semantic checks are made to succeed unconditionally.
  return RunPrescan() && RunParse() && (RunSemanticChecks() || true);
}

bool CodeGenAction::BeginSourceFileAction() {
  llvmCtx_ = std::make_unique<llvm::LLVMContext>();
  if (this->currentInput().kind().GetLanguage() == Language::LLVM_IR) {
    // Parse the bitcode...
    SMDiagnostic err;
    llvmModule_ = parseIRFile(currentInput().file(), err, *llvmCtx_);

    return (nullptr != llvmModule_);
  }

  bool res = RunPrescan() && RunParse() && RunSemanticChecks();
  if (!res)
    return res;

  CompilerInstance &ci = this->instance();

  // Load the MLIR dialects required by Flang
  mlir::DialectRegistry registry;
  ci.setMlirCtx(std::make_unique<mlir::MLIRContext>(registry));
  fir::support::registerNonCodegenDialects(registry);
  fir::support::loadNonCodegenDialects(ci.mlirCtx());

  // Create a LoweringBridge
  auto &defKinds = ci.invocation().semanticsContext().defaultKinds();
  fir::KindMapping kindMap(&ci.mlirCtx(),
      llvm::ArrayRef<fir::KindTy>{fir::fromDefaultKinds(defKinds)});
  auto lb = Fortran::lower::LoweringBridge::create(ci.mlirCtx(), defKinds,
      ci.invocation().semanticsContext().intrinsics(), ci.parsing().allCooked(),
      "", kindMap);

  // Create a parse tree and lower it to FIR
  auto &parseTree{*ci.parsing().parseTree()};
  lb.lower(parseTree, ci.invocation().semanticsContext());
  ci.setMlirModule(std::make_unique<mlir::ModuleOp>(lb.getModule()));

  // Run the default passes.
  mlir::PassManager pm(&ci.mlirCtx(), mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

  if (mlir::failed(pm.run(ci.mlirModule()))) {
    unsigned diagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "verification of lowering to FIR failed");
    ci.diagnostics().Report(diagID);
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Custom ExecuteAction
//===----------------------------------------------------------------------===//
void InputOutputTestAction::ExecuteAction() {
  CompilerInstance &ci = instance();

  // Create a stream for errors
  std::string buf;
  llvm::raw_string_ostream error_stream{buf};

  // Read the input file
  Fortran::parser::AllSources &allSources{ci.allSources()};
  std::string path{GetCurrentFileOrBufferName()};
  const Fortran::parser::SourceFile *sf;
  if (path == "-")
    sf = allSources.ReadStandardInput(error_stream);
  else
    sf = allSources.Open(path, error_stream, std::optional<std::string>{"."s});
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the contents of the input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Copy the contents from the input file to the output file
  if (!ci.IsOutputStreamNull()) {
    // An output stream (outputStream_) was set earlier
    ci.WriteOutputStream(fileContent.data());
  } else {
    // No pre-set output stream - create an output file
    os = ci.CreateDefaultOutputFile(
        /*binary=*/true, GetCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  }
}

void PrintPreprocessedAction::ExecuteAction() {
  std::string buf;
  llvm::raw_string_ostream outForPP{buf};

  // Format or dump the prescanner's output
  CompilerInstance &ci = this->instance();
  if (ci.invocation().preprocessorOpts().noReformat) {
    ci.parsing().DumpCookedChars(outForPP);
  } else {
    ci.parsing().EmitPreprocessedSource(
        outForPP, !ci.invocation().preprocessorOpts().noLineDirectives);
  }

  // Print diagnostics from the prescanner
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.IsOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.WriteOutputStream(outForPP.str());
    return;
  }

  // Create a file and save the preprocessed output there
  std::unique_ptr<llvm::raw_pwrite_stream> os{ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName())};
  if (!os) {
    return;
  }

  (*os) << outForPP.str();
}

void DebugDumpProvenanceAction::ExecuteAction() {
  this->instance().parsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::ExecuteAction() {
}

void DebugUnparseNoSemaAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);
}

void DebugUnparseAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  CompilerInstance &ci = this->instance();
  auto os{ci.CreateDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName())};

  // TODO: Options should come from CompilerInvocation
  Unparse(*os, *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugUnparseWithSymbolsAction::ExecuteAction() {
  auto &parseTree{*instance().parsing().parseTree()};

  Fortran::semantics::UnparseWithSymbols(
      llvm::outs(), parseTree, /*encoding=*/Fortran::parser::Encoding::UTF_8);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugDumpSymbolsAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  auto &semantics = ci.semantics();

  auto tables{Fortran::semantics::BuildRuntimeDerivedTypeTables(
      instance().invocation().semanticsContext())};
  // The runtime derived type information table builder may find and report
  // semantic errors. So it is important that we report them _after_
  // BuildRuntimeDerivedTypeTables is run.
  reportFatalSemanticErrors();

  if (!tables.schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
  }

  // Dump symbols
  semantics.DumpSymbols(llvm::outs());
}

void DebugDumpAllAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Dump parse tree
  auto &parseTree{instance().parsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &ci.invocation().asFortran());

  auto &semantics = ci.semantics();
  auto tables{Fortran::semantics::BuildRuntimeDerivedTypeTables(
      instance().invocation().semanticsContext())};
  // The runtime derived type information table builder may find and report
  // semantic errors. So it is important that we report them _after_
  // BuildRuntimeDerivedTypeTables is run.
  reportFatalSemanticErrors();

  if (!tables.schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
  }

  // Dump symbols
  llvm::outs() << "=====================";
  llvm::outs() << " Flang: symbols dump ";
  llvm::outs() << "=====================\n";
  semantics.DumpSymbols(llvm::outs());
}

void DebugDumpParseTreeNoSemaAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());
}

void DebugDumpParseTreeAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugMeasureParseTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (!ci.parsing().messages().empty() &&
      (ci.invocation().warnAsErr() ||
          ci.parsing().messages().AnyFatalError())) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();

    ci.parsing().messages().Emit(
        llvm::errs(), this->instance().allCookedSources());
    return;
  }

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  auto &parseTree{*ci.parsing().parseTree()};

  // Measure the parse tree
  MeasurementVisitor visitor;
  Fortran::parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void DebugPreFIRTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  auto &parseTree{*ci.parsing().parseTree()};

  // Dump pre-FIR tree
  if (auto ast{Fortran::lower::createPFT(
          parseTree, ci.invocation().semanticsContext())}) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.diagnostics().Report(diagID);
  }
}

void DebugDumpParsingLogAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  ci.parsing().Parse(llvm::errs());
  ci.parsing().DumpParsingLog(llvm::outs());
}

void GetDefinitionAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  parser::AllCookedSources &cs = ci.allCookedSources();
  unsigned diagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Symbol not found");

  auto gdv = ci.invocation().frontendOpts().getDefVals;
  auto charBlock{cs.GetCharBlockFromLineAndColumns(
      gdv.line, gdv.startColumn, gdv.endColumn)};
  if (!charBlock) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "String range: >" << charBlock->ToString() << "<\n";

  auto *symbol{ci.invocation()
                   .semanticsContext()
                   .FindScope(*charBlock)
                   .FindSymbol(*charBlock)};
  if (!symbol) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";

  auto sourceInfo{cs.GetSourcePositionRange(symbol->name())};
  if (!sourceInfo) {
    llvm_unreachable(
        "Failed to obtain SourcePosition."
        "TODO: Please, write a test and replace this with a diagnostic!");
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";
  llvm::outs() << symbol->name().ToString() << ": "
               << sourceInfo->first.file.path() << ", "
               << sourceInfo->first.line << ", " << sourceInfo->first.column
               << "-" << sourceInfo->second.column << "\n";
}

void GetSymbolsSourcesAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  ci.semantics().DumpSymbolsSources(llvm::outs());
}

void EmitMLIRAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Print the output. If a pre-defined output stream exists, dump the MLIR
  // content there.
  if (!ci.IsOutputStreamNull()) {
    ci.mlirModule().print(ci.GetOutputStream());
    return;
  }

  // ... otherwise, print to a file.
  auto os{ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "mlir")};
  if (!os) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the output file");
    ci.diagnostics().Report(diagID);
    return;
  }

  ci.mlirModule().print(*os);
}

#include "flang/Tools/CLOptions.inc"

// Lower the previously generated MLIR module into an LLVM IR module
void CodeGenAction::GenerateLLVMIR() {
  CompilerInstance &ci = this->instance();
  auto mlirMod = ci.mlirModule();

  auto &ctx = ci.mlirCtx();
  fir::support::loadDialects(ctx);
  fir::support::registerLLVMTranslation(ctx);

  // Set-up the MLIR pass manager
  fir::setTargetTriple(mlirMod, ci.invocation().targetOpts().triple);
  auto &defKinds = ci.invocation().semanticsContext().defaultKinds();
  fir::KindMapping kindMap(&ci.mlirCtx(),
      llvm::ArrayRef<fir::KindTy>{fir::fromDefaultKinds(defKinds)});
  fir::setKindMapping(mlirMod, kindMap);
  mlir::PassManager pm(&ci.mlirCtx(), mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Create the pass pipeline
  fir::createMLIRToLLVMPassPipeline(pm);

  // Run the pass manager
  if (!mlir::succeeded(pm.run(mlirMod))) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Lowering to LLVM IR failed");
    ci.diagnostics().Report(diagID);
  }

  // Translate to LLVM IR
  auto optName = mlirMod.getName();
  llvmModule_ = mlir::translateModuleToLLVMIR(
      mlirMod, *llvmCtx_, optName ? *optName : "FIRModule");

  if (!llvmModule_) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the LLVM module");
    ci.diagnostics().Report(diagID);
    return;
  }
}

void EmitLLVMAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule_)
    GenerateLLVMIR();

  // Print the generated LLVM IR. If there is no pre-defined output stream to
  // print to, create an output file.
  std::unique_ptr<llvm::raw_ostream> os;
  if (ci.IsOutputStreamNull()) {
    os = ci.CreateDefaultOutputFile(
        /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "ll");
    if (!os) {
      unsigned diagID = ci.diagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "failed to create the output file");
      ci.diagnostics().Report(diagID);
      return;
    }
  }

  if (!ci.IsOutputStreamNull()) {
    llvmModule_->print(
        ci.GetOutputStream(), /*AssemblyAnnotationWriter=*/nullptr);
  } else {
    llvmModule_->print(*os, /*AssemblyAnnotationWriter=*/nullptr);
  }
}

void EmitLLVMBitcodeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule_)
    GenerateLLVMIR();

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;

  // Create `Target`
  std::string error;
  std::string theTriple = llvmModule_->getTargetTriple();
  const llvm::Target *theTarget =
      TargetRegistry::lookupTarget(theTriple, error);
  assert(theTarget && "Failed to create Target");

  // Create `TargetMachine`
  std::unique_ptr<TargetMachine> TM;

  TM.reset(theTarget->createTargetMachine(
      theTriple, /*CPU=*/"", /*Features=*/"", llvm::TargetOptions(), None));
  llvmModule_->setDataLayout(TM->createDataLayout());
  assert(TM && "Failed to create TargetMachine");

  PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);

  // Generate an output file
  std::unique_ptr<llvm::raw_ostream> os = ci.CreateDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "bc");
  if (!os) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the output file");
    ci.diagnostics().Report(diagID);
    return;
  }

  MPM.addPass(BitcodeWriterPass(*os));
  MPM.run(*llvmModule_, MAM);
}

void BackendAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule_)
    GenerateLLVMIR();

  // Create `Target`
  std::string error;
  std::string theTriple = llvmModule_->getTargetTriple();
  const llvm::Target *theTarget =
      TargetRegistry::lookupTarget(theTriple, error);
  assert(theTarget && "Failed to create Target");

  // Create `TargetMachine`
  std::unique_ptr<TargetMachine> TM;
  TM.reset(theTarget->createTargetMachine(
      theTriple, /*CPU=*/"", /*Features=*/"", llvm::TargetOptions(), None));
  llvmModule_->setDataLayout(TM->createDataLayout());
  assert(TM && "Failed to create TargetMachine");

  // If the output stream is a file, generate it and define the corresponding
  // output stream. If a pre-defined output stream is available, we will use
  // that instead.
  //
  // NOTE: `os` is a smart pointer that will be destroyed at the end of this
  // method. However, it won't be written to until `CodeGenPasses` is
  // destroyed. By defining `os` before `CodeGenPasses`, we make sure that the
  // output stream won't be destroyed before it is written to. This only
  // applies when an output file is used (i.e. there is no pre-defined output
  // stream).
  // TODO: Revisit once the new PM is ready (i.e. when `CodeGenPasses` is
  // updated to use it).
  std::unique_ptr<llvm::raw_pwrite_stream> os;
  if (ci.IsOutputStreamNull()) {
    // Get the output buffer/file
    switch (_act) {
    case BackendAct::Backend_EmitAssembly:
      os = ci.CreateDefaultOutputFile(
          /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName(), "s");
      break;
    case BackendAct::Backend_EmitObj:
      os = ci.CreateDefaultOutputFile(
          /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName(), "o");
      break;
    }
    if (!os) {
      unsigned diagID = ci.diagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "failed to create the output file");
      ci.diagnostics().Report(diagID);
      return;
    }
  }

  // Create an LLVM code-gen pass pipeline. Currently only the legacy pass
  // manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  legacy::PassManager CodeGenPasses;
  CodeGenPasses.add(createTargetTransformInfoWrapperPass(TargetIRAnalysis()));
  Triple triple(llvmModule_->getTargetTriple());
  std::unique_ptr<llvm::TargetLibraryInfoImpl> TLII =
      std::make_unique<llvm::TargetLibraryInfoImpl>(triple);
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(*TLII));

  llvm::CodeGenFileType cgft = (_act == BackendAct::Backend_EmitAssembly)
      ? llvm::CodeGenFileType::CGFT_AssemblyFile
      : llvm::CodeGenFileType::CGFT_ObjectFile;
  if (TM->addPassesToEmitFile(CodeGenPasses,
          ci.IsOutputStreamNull() ? *os : ci.GetOutputStream(), nullptr,
          cgft)) {
    unsigned diagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "emission of this file type is not supported");
    ci.diagnostics().Report(diagID);
    return;
  }

  // Run the code-gen passes
  CodeGenPasses.run(*llvmModule_);
}

void InitOnlyAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID =
      ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Warning,
          "Use `-init-only` for testing purposes only");
  ci.diagnostics().Report(DiagID);
}

void PluginParseTreeAction::ExecuteAction() {}
