#ifndef FRONTEND_AST_H
#define FRONTEND_AST_H

#include <map>


#include "ast.h"
#include "mlir/IR/MLIRContext.h"

enum Token {
    tok_eof = -1,

    // Commands
    tok_def = -2,

    tok_identifier = -3,
    tok_number = -4,
};

class Parser {

    enum Token {
        tok_eof = -1,
        tok_def = -2,
        tok_identifier = -3,
        tok_number=-4,
    };

    mlir::MLIRContext Context;
    std::string IdentifierStr;
    double NumVal;
    std::map<char, int> BinopPrecedence;
    int CurrentToken;

     
    int GetTokPrecedence();    
    int gettok();
    int getNextToken();

    std::unique_ptr<ExprAST> ParseParenExpr();
    std::unique_ptr<VariableExprAST> ParseIdentifierExpr();
    std::unique_ptr<ExprAST> ParseNumberExpr();
    std::unique_ptr<ExprAST> ParseBinOpRHS(int ExpressionPrecedence, std::unique_ptr<ExprAST> LHS);
    std::unique_ptr<ExprAST> ParseExpression();
    std::unique_ptr<GroupPrototypeAST>ParseGroupPrototype();
    std::unique_ptr<ExprAST> ParseGroup();
    std::unique_ptr<ExprAST> ParsePrimary();
    std::unique_ptr<ExprAST> ParseAssign();

public:
    Parser(): BinopPrecedence {
        {'+', 10},
        {'-', 10},
        {'*', 30},
    } {};
    int parse();
};

#endif // FRONTEND_AST_H
