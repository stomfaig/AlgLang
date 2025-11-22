#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"
#include "Alg/AlgDialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::alg;

#define GET_OP_CLASSES
#include "Alg/AlgOps.cpp.inc"


llvm::LogicalResult AddOp::verify() {
    auto elem1 = getElem1();
    auto elem2 = getElem2();

    auto elem1Type = mlir::dyn_cast<ElementType>(elem1.getType());
    auto elem2Type = mlir::dyn_cast<ElementType>(elem2.getType());

    if (elem1Type.getGroup() != elem2Type.getGroup()) {
        return llvm::failure();
    }

    return llvm::success();
}