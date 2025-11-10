#ifndef FRONTEND_MLIRGEN_CPP
#define FRONTEND_MLIRGEN_CPP

#include <map>

#include "mlirgen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "Alg/AlgDialect.h"
#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"

#include "ast.h"

mlir::LogicalResult MLIRGenImpl::declare(llvm::StringRef varname, mlir::Value varval) {
    if (SymbolTable.count(varname))
        return mlir::failure();
    SymbolTable.insert({varname, varval});
    return mlir::success();
}

mlir::Value MLIRGenImpl::mlirGen(const ExprAST &expr) {
    // Not implemented.
    return nullptr;
}

mlir::Value MLIRGenImpl::mlirGen(const AssignAST &assign) {

    mlir::Value Val = mlirGen(assign.getRHS());

    if (failed(declare(assign.getLHS().getName(), Val))) {
        return nullptr;
    }

    return Val;
}

mlir::ArrayAttr makeIntArrayAttr(mlir::OpBuilder &builder,
                                 const std::vector<int64_t> &v) {
  llvm::SmallVector<mlir::Attribute, 8> elems;
  elems.reserve(v.size());
  for (int64_t x : v)
    elems.push_back(builder.getI64IntegerAttr(x)); // integer attrs
  return builder.getArrayAttr(elems);
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

        mlir::Type elemType = mlir::IntegerType::get(Context, 64);
        
        llvm::SmallVector<int64_t, 1> shapeVec = {static_cast<int64_t>(N)};
        llvm::ArrayRef<int64_t> shape(shapeVec);

        auto Vector_Type = mlir::VectorType::get(shape, elemType);

        auto Type = mlir::alg::ElementType::get(Context, Group, Vector_Type);

        auto CreateOp = mlir::alg::ElementCreateOp::create(Builder, Loc, Type, arrAttr, Group);
        CreateOp.print(llvm::errs());
        llvm::errs() << "\n";
        CreateOp.getResult();

        idx++;
    }

    // return a vector of all the emitted values.
    return nullptr;
}

mlir::Value MLIRGenImpl::mlirGen(const BinaryOpAST &binop) {

    auto Loc = Builder.getUnknownLoc();

    auto LHS = mlirGen(binop.getLHS());
    auto RHS = mlirGen(binop.getRHS());

    // TODO: need to check that the two types actually agree, but maybe this can be done in MLIR.

    auto AddOp = mlir::alg::AddOp::create(Builder, Loc, LHS.getType(), LHS, RHS);

    return AddOp.getResult();
}

#endif // FRONTEND_MLIRGEN_CPP
