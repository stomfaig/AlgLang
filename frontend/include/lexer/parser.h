#ifndef FRONTEND_AST_H
#define FRONTEND_AST_H

#include "ast.h"

enum Token {
    tok_eof = -1,

    // Commands
    tok_def = -2,

    tok_identifier = -3,
    tok_number = -4,
};

int run();

static std::unique_ptr<ExprAST> ParseParenExpr();

static std::unique_ptr<ExprAST> ParseIdentifierExpr();

static std::unique_ptr<ExprAST> ParseNumberExpr();

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExpressionPrecedence, std::unique_ptr<ExprAST> LHS);

static std::unique_ptr<ExprAST> ParseExpression();

static std::unique_ptr<GroupPrototypeAST>ParseGroupPrototype();

static std::unique_ptr<GroupAST> ParseGroup();

static std::unique_ptr<ExprAST> ParsePrimary();

#endif // FRONTEND_AST_H