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
    MLIRGenImpl(mlir::MLIRContext &context);
    void mlirModuleGen(const ExprAST &expr);
    void dumpModule() {
        Module->dump();
    }

private: 

mlir::ModuleOp Module;
mlir::OpBuilder Builder;
std::map<std::string, mlir::Value> SymbolTable;
    mlir::LogicalResult declare(std::string varname, mlir::Value varval);
    mlir::Value mlirGen(const ExprAST &expr);
    mlir::Value mlirGen(const NumberExprAST &numexpr);
    mlir::Value mlirGen(const VariableExprAST &var);
    mlir::Value mlirGen(const AssignAST &assign);
    mlir::Value mlirGen(const GroupAST &group);
    mlir::Value mlirGen(const BinaryOpAST &binop);

};

#endif // FRONTEND_MLIRGEN_H