#include "driver.h"


int main(int argc, char* argv[]) {
    auto Options = CLIParser(argc, argv);
    AlgDriver Driver = AlgDriver(Options);

    return Driver.run();
}