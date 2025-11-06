cmake_minimum_required(VERSION 3.20.0)
project(AlgLang)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

# Define the default argument to use by `lit` when testing.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

set(LLVM_INCLUDE_DIRS
    "${LLVM_DIR}/llvm/include"
    "${LLVM_DIR}/build/include"
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(alg_dialect)
if(FRONTEND)
    message(STATUS "Compiling compiler frontend.")
    add_subdirectory(frontend)
endif()