#ifndef FRONTEND_MLIRGEN_H
#define FRONTEND_MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"

#include "ast.h"
#include <string>
#include <utility>

class MLIRGenImpl {
    
public:
    MLIRGenImpl(mlir::MLIRContext &context);
    void mlirModuleGen(const ExprAST &expr);
    void dumpModule() {
        Module->dump();
    }
    mlir::OwningOpRef<mlir::ModuleOp> getModule() {
        return std::move(Module);
    }
    void mlirGen(Program &program);

private: 

// For the duration of building the module, we keep it in the code generator.
mlir::OwningOpRef<mlir::ModuleOp> Module;
mlir::OpBuilder Builder;
std::map<std::pair<std::string, std::string>, mlir::Value> SymbolTable;
std::map<const VariableExprAST*, std::string> VariableGroups;
    mlir::LogicalResult declare(std::string groupname, std::string varname, mlir::Value varval);
    mlir::Value mlirGen(const ExprAST &expr);
    mlir::Value mlirGen(const NumberExprAST &numexpr);
    mlir::Value mlirGen(const VariableExprAST &var);
    mlir::Value mlirGen(const AssignAST &assign);
    mlir::Value mlirGen(const GroupAST &group);
    mlir::Value mlirGen(const BinaryOpAST &binop);
    mlir::Value mlirGen(const ConstrAST &constraint);
};

#endif // FRONTEND_MLIRGEN_H