add_library(Lower)

target_sources(Lower
    PRIVATE
        lower.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ../../include/lower
        FILES
            ../../include/lower/lower.h
)

target_link_libraries(Lower
    MLIRAlg   
)