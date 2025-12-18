add_library(Parser)

target_sources(Parser
    PRIVATE
        parser.cpp
    
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/parser
        FILES
            ../../include/parser/parser.h
)

target_link_libraries(Parser
    MLIRAlg
    AST
)
