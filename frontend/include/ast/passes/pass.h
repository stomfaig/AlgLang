#ifndef FRONTEND_AST_PASS_H
#define FRONTEND_AST_PASS_H

#include "ast.h"

class ASTPass : public ASTVisitor {
public:
    virtual std::string name() const = 0;
    virtual void run(Program &program) {
        for (auto &root : program.getTopLevelNodes()) {
            root->accept(*this);
        }
    }
};

#endif // FRONTEND_AST_PASS_H