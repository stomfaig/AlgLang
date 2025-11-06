add_executable(Compiler)

target_sources(Compiler
    PRIVATE
        main.cpp
)

target_link_libraries(Compiler
    PRIVATE
        MLIRAlg
        Lexer
)

add_subdirectory(lib)