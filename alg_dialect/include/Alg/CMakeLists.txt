set(LLVM_TARGET_DEFINITIONS Ops.td)

set(MLIR_INCLUDE_DIRS
  ${MLIR_INCLUDE_DIR}               # e.g., llvm-project/build/lib/include
  ${MLIR_BINARY_DIR}/include        # generated includes
)

mlir_tablegen(AlgOps.h.inc -gen-op-decls -dialect=alg)
mlir_tablegen(AlgOps.cpp.inc -gen-op-defs -dialect=alg)
mlir_tablegen(AlgDialect.h.inc -gen-dialect-decls -dialect=alg)
mlir_tablegen(AlgDialect.cpp.inc -gen-dialect-defs -dialect=alg)
mlir_tablegen(AlgTypes.h.inc -gen-typedef-decls -dialect=alg --typedefs-dialect=alg)
mlir_tablegen(AlgTypes.cpp.inc -gen-typedef-defs -dialect=alg --typedefs-dialect=alg)
mlir_tablegen(AlgAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=alg)
mlir_tablegen(AlgAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=alg)

add_public_tablegen_target(MLIRAlgOpsIncGen)