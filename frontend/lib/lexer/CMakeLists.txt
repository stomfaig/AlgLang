add_library(Lexer)

target_sources(Lexer
    PRIVATE
        parser.cpp
        ast.cpp
    
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/lexer
        FILES
            ../../include/lexer/ast.h
            ../../include/lexer/parser.h
)