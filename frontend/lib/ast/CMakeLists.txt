add_library(AST)

target_sources(AST
    PRIVATE
        ast.cpp
    
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/ast
        FILES
            ../../include/ast/ast.h
)