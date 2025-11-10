add_library(Codegen)

target_sources(Codegen
    PRIVATE
        mlirgen.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/mlirgen
        FILES
            ../../include/mlirgen/mlirgen.h
)

target_link_libraries(Codegen
    PRIVATE
        MLIRAlg
        AST
)