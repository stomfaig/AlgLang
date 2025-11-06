#include "Alg/AlgDialect.h"
#include "Alg/AlgOps.h"
#include "Alg/AlgTypes.h"

using namespace mlir;
using namespace mlir::alg;

#include "Alg/AlgDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void AlgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Alg/AlgOps.cpp.inc"
      >();
  registerTypes();
}