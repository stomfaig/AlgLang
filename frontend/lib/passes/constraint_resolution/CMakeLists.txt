add_library(ConstraintResolution)

target_sources(ConstraintResolution
    PRIVATE
        constraint_resolution.cpp
        constraints.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../../include/passes/constraint_resolution
        FILES
            ../../../include/passes/constraint_resolution/constraint_resolution.h
            ../../../include/passes/constraint_resolution/constraints.h
)

target_link_libraries(ConstraintResolution
    PRIVATE
        MLIRAlg
)