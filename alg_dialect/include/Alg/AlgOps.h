#ifndef ALG_ALGOPS_H
#define ALG_ALGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Alg/AlgTypes.h"

#define GET_OP_CLASSES
#include "Alg/AlgOps.h.inc"

#endif // ALG_ALGOPS_H