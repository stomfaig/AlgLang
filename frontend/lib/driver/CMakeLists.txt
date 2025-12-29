add_library(Driver)

target_sources(Driver
    PRIVATE
        driver.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/driver
        FILES
            ../../include/driver/driver.h
)

target_link_libraries(Driver
    PRIVATE     
        Lower
        MLIRAlg
        AST
        Codegen
        Parser
        ConstraintResolution
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRRewrite
        MLIRDialectUtils
        MLIRVectorDialect
        MLIRArithDialect
)