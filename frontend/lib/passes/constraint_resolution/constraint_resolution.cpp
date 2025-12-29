#ifndef ALG_FRONTEND_CONSTRAINT_RESOLUTION_CPP
#define ALG_FRONTEND_CONSTRAINT_RESOLUTION_CPP


#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include <csignal>
#include <cstddef>
#include <memory>
#include <string>

#include "constraint_resolution.h"
#include "constraints.h"

AlgVec getExprValue(mlir::Operation& op);

AlgVec getExprValue(mlir::alg::AddOp add) {
    auto term1 = add.getElem1();
    auto acc = getExprValue(*term1.getDefiningOp());
    auto term2 = add.getElem2();
    acc += getExprValue(*term2.getDefiningOp());
    return acc;
}

AlgVec getExprValue(mlir::alg::UnaryNeg& un_neg) {
    auto value = un_neg->getOperand(0);
    auto op = value.getDefiningOp();

    auto acc = getExprValue(*op);
    acc *= -1;
    return acc;
}

AlgVec getExprValue(mlir::alg::IntMul& int_mul) {
    auto value = int_mul->getOperand(0);
    auto op = value.getDefiningOp();

    auto acc = getExprValue(*op);
    return acc;
}

AlgVec getExprValue(mlir::alg::ElementCreateOp& create) {
    auto attr = create->getAttrOfType<mlir::ArrayAttr>("data");
    auto vec =  AlgVec(attr.size());
    size_t i = 0;
    for (auto elem : attr.getValue()) {
        auto intAttr = dyn_cast<mlir::IntegerAttr>(elem);
        int64_t value = intAttr.getInt();
        vec[i++] = value;
    }
    return vec;
}   

AlgVec getExprValue(mlir::Operation& op) {

    if (auto add = dyn_cast<mlir::alg::AddOp>(op)) {
        return getExprValue(add);
    } else if (auto un_neg = dyn_cast<mlir::alg::UnaryNeg>(op)) {
        return getExprValue(un_neg);
    } else if (auto int_mul = dyn_cast<mlir::alg::IntMul>(op)) {
        return getExprValue(int_mul);
    } else if (auto create = dyn_cast<mlir::alg::ElementCreateOp>(op)) {
        return getExprValue(create);
    } else {
        /// TODO: Raise error.
    }

    return AlgVec(0);
}

void ConstrResPass::runOnOperation() {

    auto module = getOperation();
    auto &region = module.getBodyRegion();

    std::map<std::string, std::unique_ptr<ConstraintTable>> constraint_tables;

    for (auto &block: region.getBlocks()) {
        for (auto &op : block.getOperations()) {
            if (auto c = dyn_cast<mlir::alg::UnaryConstrOp>(op)) {
                auto op_type = dyn_cast<mlir::alg::ElementType>(op.getOperandTypes()[0]);
                if (!op_type)
                    signalPassFailure();
        
                auto gp_attr = op_type.getGroup();
                auto group_name = gp_attr.getName().str();
                
                if (!constraint_tables.count(group_name))
                    constraint_tables.insert({group_name, std::make_unique<ConstraintTable>(vec.size())});

                auto constr_expr_op = c.getElem().getDefiningOp();
                auto vec = getExprValue(*constr_expr_op);

                auto search = constraint_tables.find(group_name);
                auto &table = search->second;

                table->introduceConstraint(vec);
            }
        }
    }
}

#endif // ALG_FRONTEND_CONSTRAINT_RESOLUTION_CPP