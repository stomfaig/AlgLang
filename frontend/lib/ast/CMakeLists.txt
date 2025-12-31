add_library(AST)

target_sources(AST
    PRIVATE
        ast.cpp
        passes/pass.cpp
        passes/group_resolution_pass.cpp
    
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/ast
        FILES
            ../../include/ast/ast.h
            ../../include/ast/passes/pass.h
            ../../include/ast/passes/group_resolution_pass.h
)