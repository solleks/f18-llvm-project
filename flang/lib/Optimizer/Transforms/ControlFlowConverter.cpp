//===-- ControlFlowConverter.cpp - convert high-level control flow --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert affine dialect operations to loop/standard dialect operations.
// Also convert the fir.select_type Op to more primitive operations.
//
// TODO: this needs either a deeper understanding of how types will be
// represented by F18 or at least a couple of runtime calls to be completed.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "flang-cfg-converter"

static llvm::cl::opt<bool> disableControlFlowLowering(
    "disable-control-flow-lowering",
    llvm::cl::desc("disable the pass to convert fir.select_type and affine "
                   "dialect operations to more primitive operations"),
    llvm::cl::init(false), llvm::cl::Hidden);

using OperandTy = llvm::ArrayRef<mlir::Value>;
using namespace fir;

namespace {

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConversionPattern {
public:
  explicit FIROpConversion(mlir::MLIRContext *ctx)
      : ConversionPattern(FromOp::getOperationName(), 1, ctx) {}

  static Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                            Block *insertBefore) {
    assert(insertBefore && "expected valid insertion block");
    return rewriter.createBlock(insertBefore->getParent(),
                                mlir::Region::iterator(insertBefore));
  }
};

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selectType = mlir::cast<SelectTypeOp>(op);
    auto conds = selectType.getNumConditions();
    auto attrName = SelectTypeOp::getCasesAttr();
    auto caseAttr = selectType->getAttrOfType<mlir::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Selector must be of type !fir.box<T>
    auto selector = selectType.getSelector(operands);
    auto loc = selectType.getLoc();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    for (decltype(conds) t = 0; t != conds; ++t) {
      auto *dest = selectType.getSuccessor(t);
      auto destOps = selectType.getSuccessorOperands(operands, t);
      auto &attr = cases[t];
      if (auto a = attr.dyn_cast<ExactTypeAttr>()) {
        genTypeLadderStep(loc, /*exactTest=*/true, selector, a.getType(), dest,
                          destOps, mod, rewriter);
        continue;
      }
      if (auto a = attr.dyn_cast<SubclassAttr>()) {
        genTypeLadderStep(loc, /*exactTest=*/false, selector, a.getType(), dest,
                          destOps, mod, rewriter);
        continue;
      }
      assert(attr.isa<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<mlir::BranchOp>(
          selectType, dest, mlir::ValueRange{destOps.getValue()});
    }
    return success();
  }

  static void genTypeLadderStep(mlir::Location loc, bool exactTest,
                                mlir::Value selector, mlir::Type ty,
                                mlir::Block *dest,
                                llvm::Optional<OperandTy> destOps,
                                mlir::ModuleOp module,
                                mlir::ConversionPatternRewriter &rewriter) {
    mlir::Type tydesc = TypeDescType::get(ty);
    auto tyattr = mlir::TypeAttr::get(ty);
    mlir::Value t = rewriter.create<GenTypeDescOp>(loc, tydesc, tyattr);
    mlir::Type selty = BoxType::get(rewriter.getNoneType());
    mlir::Value csel = rewriter.create<ConvertOp>(loc, selty, selector);
    mlir::Type tty = ReferenceType::get(rewriter.getNoneType());
    mlir::Value ct = rewriter.create<ConvertOp>(loc, tty, t);
    std::vector<mlir::Value> actuals = {csel, ct};
    auto fty = rewriter.getI1Type();
    std::vector<mlir::Type> argTy = {selty, tty};
    llvm::StringRef funName =
        exactTest ? "FIXME_exact_type_match" : "FIXME_isa_type_test";
    createFuncOp(rewriter.getUnknownLoc(), module, funName,
                 rewriter.getFunctionType(argTy, fty));
    // FIXME: need to call actual runtime routines for (1) testing if the
    // runtime type of the selector is an exact match to a derived type or (2)
    // testing if the runtime type of the selector is a derived type or one of
    // that derived type's subtypes.
    auto cmp = rewriter.create<fir::CallOp>(
        loc, fty, mlir::SymbolRefAttr::get(rewriter.getContext(), funName),
        actuals);
    auto *thisBlock = rewriter.getInsertionBlock();
    auto *newBlock = createBlock(rewriter, dest);
    rewriter.setInsertionPointToEnd(thisBlock);
    if (destOps.hasValue())
      rewriter.create<mlir::CondBranchOp>(loc, cmp.getResult(0), dest,
                                          destOps.getValue(), newBlock,
                                          llvm::None);
    else
      rewriter.create<mlir::CondBranchOp>(loc, cmp.getResult(0), dest,
                                          newBlock);
    rewriter.setInsertionPointToEnd(newBlock);
  }
};

/// Convert affine dialect, fir.select_type to standard dialect
class ControlFlowLoweringPass
    : public ControlFlowLoweringBase<ControlFlowLoweringPass> {
public:
  explicit ControlFlowLoweringPass() {}

  void runOnFunction() override {
    if (disableControlFlowLowering)
      return;
    auto &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    patterns.insert<SelectTypeOpConversion>(&context);
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<fir::FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<fir::SelectTypeOp>();

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createControlFlowLoweringPass() {
  return std::make_unique<ControlFlowLoweringPass>();
}
