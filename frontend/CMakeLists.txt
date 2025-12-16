add_executable(Compiler)

target_sources(Compiler
    PRIVATE
        main.cpp
)

target_link_libraries(Compiler
    PRIVATE
        Driver
        Parser
        Codegen
)

add_subdirectory(lib)