#ifndef ALG_FRONTEND_LOWER_H
#define ALG_FRONTEND_LOWER_H

#include <memory>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlirgen.h"
#include "parser.h"

struct CompilerOptions {
  std::string inputFile;
  bool emitMLIR;
  bool runCanonicalization;
  std::string passPipeline;
};

CompilerOptions CLIParser(int argc, char **argv);

/// Driver class for compiling alg.
class AlgDriver {

private:
    CompilerOptions Options;
    mlir::MLIRContext Context;
    Parser Parser;
    mlir::PassManager Manager;
    MLIRGenImpl Implementor;

    void setupPasses();


public:
    AlgDriver(CompilerOptions &Options);
    int run();
};



#endif // ALG_FRONTEND_LOWER_H