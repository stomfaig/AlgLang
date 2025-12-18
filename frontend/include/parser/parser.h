#ifndef FRONTEND_AST_H
#define FRONTEND_AST_H

#include <map>
#include <fstream>
#include <iostream>

#include "ast.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"

enum Token {
    tok_eof = -1,

    // Commands
    tok_def = -2,

    tok_identifier = -3,
    tok_number = -4,

    tok_constr=-5,
};

class Parser {

    enum Token {
        tok_eof = -1,
        tok_def = -2,
        
        tok_identifier = -3,
        tok_number=-4,
        
        tok_constr=-5,
    };
    std::map<char, int> BinopPrecedence {
        {'+', 10},
        {'-', 10},
        {'*', 30},
    };
    std::ifstream Stream;
    std::string IdentifierStr;
    double NumVal;
    int CurrentToken;
    mlir::MLIRContext &Context;

    std::string File;
    unsigned int CurrentLine = 1;
    unsigned int CurrentChar = 0;
     
    int GetTokPrecedence();    
    int gettok();
    int getNextToken();

    char get_next_char();
    mlir::Location getLocation();

    std::unique_ptr<ExprAST> ParseParenExpr();
    std::unique_ptr<VariableExprAST> ParseIdentifierExpr();
    std::unique_ptr<ExprAST> ParseNumberExpr();
    std::unique_ptr<ExprAST> ParseBinOpRHS(int ExpressionPrecedence, std::unique_ptr<ExprAST> LHS);
    std::unique_ptr<ExprAST> ParseExpression();
    std::unique_ptr<GroupPrototypeAST>ParseGroupPrototype();
    std::unique_ptr<ExprAST> ParseGroup();
    std::unique_ptr<ExprAST> ParseConstraint();
    std::unique_ptr<ExprAST> ParsePrimary();
    std::unique_ptr<ExprAST> ParseAssign();

public:
    Parser(mlir::MLIRContext &Context, std::string File): Stream(File), Context(Context), File(File) {};

    /// This method returns a unique_ptr to the AST generated from the source
    std::vector<std::unique_ptr<ExprAST>> Parse();
};

#endif // FRONTEND_AST_H
