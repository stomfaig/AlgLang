#ifndef ALG_FRONTEND_LOWER_CPP
#define ALG_FRONTEND_LOWER_CPP

#include "driver.h"

#include "ast.h"
#include "mlirgen.h"
#include "parser.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"


AlgDriver::AlgDriver(CompilerOptions &Options):
    Options(Options),
    Context(),
    Parser(Context, Options.inputFile),
    Manager(&Context),
    Implementor(Context)
{
    // Based on the params, add passes
    //pm.addPass(std::make_unique<AlgLoweringPass>());

}

int AlgDriver::run() {
    auto AST = Parser.Parse();
    for (const std::unique_ptr<ExprAST> &root : AST) {
        root->dump();
    }
    
    //mlir::ModuleOp Module = Implementor.mlirModuleGen(*AST);
    //Module->dump();

    return 0;
}

#endif // ALG_FRONTEND_LOWER_CPP