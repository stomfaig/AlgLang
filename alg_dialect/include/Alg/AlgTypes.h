#ifndef ALG_TYPES_H
#define ALG_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Alg/AlgAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Alg/AlgTypes.h.inc"

#endif // ALG_TYPES_H


