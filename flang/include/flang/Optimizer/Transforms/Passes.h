//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class BlockAndValueMapping;
class Operation;
class Region;
} // namespace mlir

namespace fir {

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createAbstractResultOptPass();
std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass> createArrayValueCopyPass();
std::unique_ptr<mlir::Pass> createCharacterConversionPass();
std::unique_ptr<mlir::Pass> createExternalNameConversionPass();
std::unique_ptr<mlir::Pass> createControlFlowLoweringPass();
std::unique_ptr<mlir::Pass> createCSEPass();
std::unique_ptr<mlir::Pass> createFirLoopResultOptPass();
std::unique_ptr<mlir::Pass> createFirToCfgPass();
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createMemoryAllocationPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
std::unique_ptr<mlir::Pass> createSimplifyRegionLitePass();

std::unique_ptr<mlir::Pass>
createMemoryAllocationPass(bool dynOnHeap, std::size_t maxStackSize);
std::unique_ptr<mlir::Pass> createAnnotateConstantOperandsPass();

/// A pass to convert the FIR dialect from "Mem-SSA" form to "Reg-SSA" form.
/// This pass is a port of LLVM's mem2reg pass, but modified for the FIR dialect
/// as well as the restructuring of MLIR's representation to present PHI nodes
/// as block arguments.
/// TODO: This pass needs some additional work.
std::unique_ptr<mlir::Pass> createMemToRegPass();
std::unique_ptr<mlir::Pass> createMemoryAllocationPass();

/// Optionally force the body of a DO to execute at least once.
bool isAlwaysExecuteLoopBody();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
