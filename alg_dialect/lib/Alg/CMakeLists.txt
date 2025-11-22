add_mlir_library(MLIRAlg
  AlgDialect.cpp
  AlgOps.cpp
  AlgTypes.cpp

  DEPENDS
    MLIRAlgOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)

target_include_directories(MLIRAlg
    PUBLIC
      ../../include
      ${CMAKE_CURRENT_BINARY_DIR}/../../include
      ${MLIR_SRC_DIR}/include               
      ${MLIR_BUILD_INCLUDE_DIR}            
)