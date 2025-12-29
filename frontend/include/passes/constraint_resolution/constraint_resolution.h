#ifndef ALG_FRONTEND_CONSTRAINT_RESOLUTION_H
#define ALG_FRONTEND_CONSTRAINT_RESOLUTION_H

#include "mlir/IR/MLIRContext.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Pass/Pass.h"

#include "Alg/AlgDialect.h"

struct ConstrResPass : public mlir::PassWrapper<ConstrResPass, mlir::OperationPass<mlir::ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstrResPass)

  mlir::StringRef getArgument() const override { return "constr-res"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::alg::AlgDialect>();
  }

  void handleConstrOp();

public:
  void runOnOperation() final;
};

#endif // ALG_FRONTEND_CONSTRAINT_RESOLUTION_H