#ifndef ALG_FRONTEND_LOWER_H
#define ALG_FRONTEND_LOWER_H

#include "mlir/IR/MLIRContext.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "Alg/AlgDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Pass/Pass.h"


struct AlgLoweringPass : public mlir::PassWrapper<AlgLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlgLoweringPass)

  mlir::StringRef getArgument() const override { return "alg-to-mlir"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::vector::VectorDialect, mlir::alg::AlgDialect>();
  }

public:
  void runOnOperation() final;
};

#endif // ALG_FRONTEND_LOWER_H