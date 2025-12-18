#ifndef ALG_FRONTEND_LOWER_CPP
#define ALG_FRONTEND_LOWER_CPP

#include "driver.h"

#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OwningOpRef.h"

#include "lower.h"
#include "ast.h"
#include "mlirgen.h"
#include "parser.h"


CompilerOptions CLIParser(int argc, char **argv) {

    bool DumpAST = false;
    bool DumpAlg = false;
    bool DumpLoweredMLIR = false;
    std::string InputFile = "";
    for (int i = 1; i < argc; i++) {
        std::string Elem = argv[i];

        if (Elem == "--dump-ast") {
            DumpAST = true;
        } else if (Elem == "--dump-alg") {
            DumpAlg = true;
        } else if (Elem == "--dump-lowered-mlir") {
            DumpLoweredMLIR = true;
        } else if (Elem == "--input") {
            InputFile = argv[++i];
        } else {
            // Implicitly, a non-qualified field will be taken as a filename too.
            if (InputFile == "")
                InputFile = Elem;
        } 
    }

    return CompilerOptions{
        InputFile,
        DumpAST,
        DumpAlg,
        DumpLoweredMLIR,
    };        
}

AlgDriver::AlgDriver(CompilerOptions &Options):
    Options(Options),
    Context(),
    Parser(Context, Options.InputFile),
    Manager(&Context),
    Implementor(Context)
{
    loadDialects();
    loadPasses();
}

void AlgDriver::loadDialects() {
    Context.getOrLoadDialect<mlir::alg::AlgDialect>();
    Context.getOrLoadDialect<mlir::arith::ArithDialect>();
}

void AlgDriver::loadPasses() {
    Manager.addPass(std::make_unique<AlgLoweringPass>());
}

int AlgDriver::run() {
    auto AST = Parser.Parse();
    
    for (const std::unique_ptr<ExprAST> &root : AST) {
        if (Options.DumpAST)
            root->dump();
        Implementor.mlirModuleGen(*root);
    }

    if (Options.DumpAlg)
        Implementor.dumpModule();

    mlir::OwningOpRef<mlir::ModuleOp> Module = Implementor.getModule();

    // TODO: fix error logging
    if (llvm::failed(Manager.run(*Module))) ;

    if (Options.DumpLoweredMLIR)
        Module->dump();

    return 0;
}

#endif // ALG_FRONTEND_LOWER_CPP