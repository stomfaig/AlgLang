#ifndef FRONTEND_MLIRGEN_CPP
#define FRONTEND_MLIRGEN_CPP

#include <map>

#include "mlirgen.h"
#include "ast.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"

#include "ast.h"

MLIRGenImpl::MLIRGenImpl(mlir::MLIRContext &context): Builder(&context) {
    Module = mlir::ModuleOp::create(Builder.getUnknownLoc());
    Builder.setInsertionPointToStart(Module.getBody());
}

mlir::ModuleOp MLIRGenImpl::mlirModuleGen(const ExprAST &expr) {
    mlirGen(expr);
    return Module;
}

/// @brief Dispatcher method based on the type of node the expression was casted from.
/// @param expr 
/// @return The generated MLIR Value.
mlir::Value MLIRGenImpl::mlirGen(const ExprAST &expr) {
    switch (expr.getKind()) {
        case ExprAST::EAK_Number:
            return mlirGen(llvm::cast<NumberExprAST>(expr));
            break;
        case ExprAST::EAK_Variable:
            return mlirGen(llvm::cast<VariableExprAST>(expr));
            break;
        case ExprAST::EAK_Assign:
            return mlirGen(llvm::cast<AssignAST>(expr));
            break;
        case ExprAST::EAK_BinaryOp:
            return mlirGen(llvm::cast<BinaryOpAST>(expr));
            break;
        case ExprAST::EAK_Group:
            return mlirGen(llvm::cast<GroupAST>(expr));
            break;
        default:
            return nullptr;
    }
} 

/// @brief Method for declaring variables in the SymbolTable.
/// @param varname Name of the variable to be the SymbolTable key.
/// @param varval MLIR Value to be associated with the variable name.
/// @return mlir::LogicalValue depending on whether there was already an item with the given key in the SymbolTable or not.
mlir::LogicalResult MLIRGenImpl::declare(std::string varname, mlir::Value varval) {
    if (SymbolTable.count(varname))
        return mlir::failure();
    SymbolTable.insert({varname, varval});
    return mlir::success();
}

/// @brief Generates MLIR for numeric constant expression, by declaring an I32 type from the arith dialect.
/// @param expr 
/// @return The numerical constant Value generated.
mlir::Value MLIRGenImpl::mlirGen(const NumberExprAST &expr) {
    auto Type = Builder.getIntegerType(32);
    auto Attr = Builder.getIntegerAttr(Type, expr.getVal());

    auto Loc = Builder.getUnknownLoc();

    auto ConstOp = mlir::arith::ConstantOp::create(Builder, Loc, Attr);

    return ConstOp.getResult();
}

/// @brief Retrieves the SSA associated with a given name.
/// @param var 
/// @return Retrieved SSA val.
mlir::Value MLIRGenImpl::mlirGen(const VariableExprAST &var) {
    // TODO: implement error logic.
    auto search = SymbolTable.find(var.getName());
    return search->second;
}

/// @brief Generates MLIR code for an assign operation, by registering the RHS SSA value into the SymbolTable (this generation might produce MLIR).
/// @param assign 
/// @return The Value of RHS.
mlir::Value MLIRGenImpl::mlirGen(const AssignAST &assign) {

    mlir::Value Val = mlirGen(assign.getRHS());

    if (failed(declare(assign.getLHS().getName(), Val))) {
        return nullptr;
    }

    return Val;
}


/// @brief Generates MLIR code for binary op into the static module, by parsing the op type, and constructing the correponding ALg ops.
/// @param binop pointer to the binop AST node.
/// @return The resulting SSA of the op.
mlir::Value MLIRGenImpl::mlirGen(const BinaryOpAST &binop) {
    
    auto Loc = Builder.getUnknownLoc();

    mlir::Value LHS = mlirGen(binop.getLHS());
    mlir::Value RHS = mlirGen(binop.getRHS());

    // Type checking is need (?)

    switch (binop.getOp()) {
        case '+': {
            auto AddOp = mlir::alg::AddOp::create(Builder, Loc, LHS.getType(), LHS, RHS);
            
            if (failed(mlir::verify(AddOp))) {
                llvm::errs() << "Invalid addition. Check group memberships.\n";
                AddOp->erase();
                return nullptr;
            }

            return AddOp.getResult();
        }
        case '-': {
            auto NegOp = mlir::alg::UnaryNeg::create(Builder, Loc, RHS.getType(), RHS);
            auto NegRHSVal = NegOp.getResult();

            auto AddOp = mlir::alg::AddOp::create(Builder, Loc, LHS.getType(), LHS, NegRHSVal);

            if (failed(mlir::verify(AddOp))) {
                llvm::errs() << "Invalid subtraction. Check group memberships.";
                NegOp->erase();
                AddOp.erase();
            }

            return AddOp.getResult();
        }
        case '*': {
            auto IntMulOp = mlir::alg::IntMul::create(Builder, Loc, LHS.getType(), LHS, RHS);
            return IntMulOp.getResult();
            break;
        }
        default:
            return nullptr;
            break;
    }

}

/// @brief Generates MLIR code for the elements of the group, by creating an Element type for each generator.
/// @param group Pointer to the GroupAST node to generate MLIR for
/// @return When done, a vector to the generated generator elements for the group.
mlir::Value MLIRGenImpl::mlirGen(const GroupAST &group) {

    auto Context = Builder.getContext();
    auto Loc = Builder.getUnknownLoc();

    mlir::StringAttr Name = Builder.getStringAttr(group.getProto().getName());

    auto Group = mlir::alg::GroupAttr::get(Context, Name);

    size_t idx = 0;
    size_t N = group.getProto().getGenerators().size();

    for (std::string GeneratorName : group.getProto().getGenerators()) {

        llvm::SmallVector<mlir::Attribute, 4> elements(N, Builder.getI32IntegerAttr(0)); 
        elements[idx] = Builder.getI32IntegerAttr(1);                                    

        mlir::ArrayAttr arrAttr = Builder.getArrayAttr(elements);

        mlir::Type elemType = mlir::IntegerType::get(Context, 32);
        
        llvm::SmallVector<int64_t, 1> shapeVec = {static_cast<int64_t>(N)};
        llvm::ArrayRef<int64_t> shape(shapeVec);

        auto Vector_Type = mlir::VectorType::get(shape, elemType);

        auto Type = mlir::alg::ElementType::get(Context, Group, Vector_Type);

        auto CreateOp = mlir::alg::ElementCreateOp::create(Builder, Loc, Type, arrAttr, Group);
        auto result = CreateOp.getResult();

        // TODO: add error handling if the registration fails
        if (failed(declare(GeneratorName, result))) {
            std::cerr << "Variable declaration failed for variable " << GeneratorName;
        }

        idx++;
    }

    // return a vector of all the emitted values.
    return nullptr;
}

#endif // FRONTEND_MLIRGEN_CPP
