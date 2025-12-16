#include "driver.h"
#include <memory>


int main(int argc, char* argv[]) {
    // For now we assume that the user only passes a file path

    CompilerOptions Options({
        argv[1],
        false,
        false,
        ""
    });
    
    AlgDriver Driver = AlgDriver(Options);

    Driver.run();
}