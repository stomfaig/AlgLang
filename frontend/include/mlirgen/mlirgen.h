#ifndef FRONTEND_MLIRGEN_H
#define FRONTEND_MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "Alg/AlgDialect.h"
#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"

#include "ast.h"

class MLIRGenImpl {
    
public:

MLIRGenImpl(mlir::MLIRContext &context) : Builder(&context) {};

//mlir::ModuleOp mlirGen(const ExprAST ModuleAST);

mlir::ModuleOp Module;
mlir::OpBuilder Builder;
std::map<llvm::StringRef, mlir::Value> SymbolTable;

mlir::Value mlirGen(const ExprAST &expr);

mlir::LogicalResult declare(llvm::StringRef varname, mlir::Value varval);

mlir::Value mlirGen(const NumberExprAST &numexpr);

mlir::Value mlirGen(const AssignAST &assign);

mlir::Value mlirGen(const GroupAST &group);

mlir::Value mlirGen(const BinaryOpAST &binop);

};

#endif // FRONTEND_MLIRGEN_H