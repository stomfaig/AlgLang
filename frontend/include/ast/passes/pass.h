#ifndef FRONTEND_AST_VISITOR_H
#define FRONTEND_AST_VISITOR_H

#include "ast.h"

class ASTPass : public ASTVisitor {
public:
    virtual std::string name() const = 0;
    virtual void run(ExprAST& root) {
        root.accept(*this);
    }
};

#endif // FRONTEND_AST_VISITOR_H