#include "Alg/AlgDialect.h"
#include "Alg/AlgTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::alg;

#define GET_TYPEDEF_CLASSES
#include "Alg/AlgTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Alg/AlgAttrs.cpp.inc"

void AlgDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Alg/AlgTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Alg/AlgAttrs.cpp.inc"
      >();
}