#ifndef FRONTEND_AST_CPP
#define FRONTEND_AST_CPP

#include "ast.h"
#include <iostream>
#include <string>


void ExprAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "ExprAST\n";
}

void NumberExprAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "NumberExpr(" << Val << ")\n";
}

void VariableExprAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "VariableExpr(" << Name << ")\n";
}

void BinaryOpAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "BinaryOp(" << Op << ")\n";
    if (LHS) LHS->dump(os, indent + 2);
    if (RHS) RHS->dump(os, indent + 2);
}

void GroupPrototypeAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "GroupPrototype(" << Name << ") [";
    for (size_t i = 0; i < Generators.size(); ++i) {
        os << Generators[i];
        if (i + 1 < Generators.size()) os << ", ";
    }
    os << "]\n";
}

void GroupAST::dump(std::ostream &os, unsigned indent) const {
    os << std::string(indent, ' ') << "GroupAST\n";
    if (Proto) Proto->dump(os, indent + 2);
    for (size_t i = 0; i < Rules.size(); ++i) {
        Rules[i]->dump(os, indent + 2);
    }
}

#endif // FRONTEND_AST_CPP