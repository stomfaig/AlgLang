#ifndef FRONTEND_PARSER_CPP
#define FRONTEND_PARSER_CPP

#include <cstddef>
#include <memory>
#include <string>
#include <iostream>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassRegistry.h"

#include "ast.h"
#include "parser.h"

int Parser::GetTokPrecedence() {
  if (!isascii(CurrentToken))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurrentToken];
  if (TokPrec <= 0) return -1;
  return TokPrec;
}

char Parser::get_next_char() {

    char Next = Stream.get();

    if (Next == '\n') {
        CurrentLine++;
        CurrentChar = 0;
    } else {
        CurrentChar++;
    }

    return Next;
}

int Parser::gettok() {
    static int LastChar = ' ';

    // Swallow spaces
    while (isspace(LastChar)) {
        LastChar = get_next_char();
    }

    // First, we check if the string starts with a number or a letter:
    // If it is a letter, it must be a command or an identifier.
    if (isalpha(LastChar)) { // The word must start with a letter
        IdentifierStr = LastChar;

        // Read all letters _and_ numbers
        while (isalnum((LastChar = get_next_char())))
            IdentifierStr += LastChar;

        // Check if we have read a command
        if (IdentifierStr == "def")
            return Token::tok_def;
        if (IdentifierStr == "constr")
            return Token::tok_constr;
        else
            return Token::tok_identifier;
    }

    // If it is a number, it must be a number
    // In our case only integers make sense.
    if (isdigit(LastChar)) {
        std::string NumString;
        NumString = LastChar;

        while(isdigit((LastChar = get_next_char()))) 
            NumString += LastChar;

        NumVal = strtod(NumString.c_str(), 0);
        return Token::tok_number;
    }

    // Finally, we handle comments:
    if (LastChar == '#') {
        // Swallow all characters until the end of the line or file
        while (LastChar != EOF || LastChar != '\n' || LastChar != '\r')
            LastChar = get_next_char();

        // If it is not the end of the file, we set LastChar to the next char,
        // and call `gettok` to return the next token.

        // NB. The tutorial does not include the `.. = getchar();` call, which seems a bit silly.
        if (LastChar != EOF)
            LastChar = get_next_char();
        
        return gettok();
    }

    // If none of the above matched, we either found EOF or a random char.
    // EOF is easy:
    if (LastChar == EOF) return Token::tok_eof;

    // Otherwise, we update LastChar, and return the ASCII code of the
    // unrecognised char:
    int RetChar = LastChar;
    LastChar = get_next_char();
    return RetChar;
}

int Parser::getNextToken() {
    CurrentToken = gettok();

    return CurrentToken;
}

mlir::Location Parser::getLocation() {
    return mlir::FileLineColLoc::get(&Context, File, CurrentLine, CurrentChar );
}

std::unique_ptr<ExprAST> Parser::ParseParenExpr() {
    // Eat `(`
    getNextToken(); 

    auto expr = ParseExpression();
    if (!expr) {
        return nullptr;
    }

    if (CurrentToken != ')') {
        // need to return an error
    }
    getNextToken(); // Eat the ')'

    return expr;
}

std::unique_ptr<VariableExprAST> Parser::ParseIdentifierExpr() {
    std::string IdentifierName = IdentifierStr;
    getNextToken();
    return std::make_unique<VariableExprAST>(IdentifierName, getLocation());
}

// This method assumes that the next token is a `Token::tok_number`
std::unique_ptr<ExprAST> Parser::ParseNumberExpr() {
    std::unique_ptr<ExprAST> result = std::make_unique<NumberExprAST>(NumVal, getLocation());
    getNextToken();
    return result;
}

std::unique_ptr<ExprAST> Parser::ParsePrimary() {
    switch (CurrentToken) {
        default:
            return nullptr;
        case Token::tok_identifier:
            return ParseIdentifierExpr();
        case Token::tok_number:
            return ParseNumberExpr();
        case '(':
            return ParseParenExpr();
    }
}

std::unique_ptr<ExprAST> Parser::ParseBinOpRHS(int ExpressionPrecedence, std::unique_ptr<ExprAST> LHS) {

    while (true) {
        int TokenPrecedence = GetTokPrecedence();

        // This implicitly also checks for invalid operations.
        if (TokenPrecedence < ExpressionPrecedence) return LHS;

        int BinaryOperation = CurrentToken;
        getNextToken();

        std::unique_ptr<ExprAST> RHS = ParsePrimary();

        if (!RHS) {
            mlir::emitError(getLocation()) << "Parsing first RHS in ParseBinOpRHS failed on token " << CurrentToken;
            return nullptr;
        }

        int NextPrecedence = GetTokPrecedence();

        if (TokenPrecedence < NextPrecedence) {
            RHS = ParseBinOpRHS(TokenPrecedence + 1, std::move(RHS));
            if (!RHS)
                return nullptr;
        }

        LHS = std::make_unique<BinaryOpAST>(BinaryOperation, std::move(LHS), std::move(RHS), getLocation());
    }
}

std::unique_ptr<ExprAST> Parser::ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS) {
        mlir::emitError(getLocation()) << "LHS parsing failed in ParseExpression.";
        return nullptr;
    }

    return ParseBinOpRHS(0, std::move(LHS));
}

std::unique_ptr<GroupPrototypeAST> Parser::ParseGroupPrototype() {

    // The parsing of the protype has the responsibility to read everything between the `def` keyword and the body of the definition.

    // This must be followed by the name of the group
    if (CurrentToken != Token::tok_identifier) {
        mlir::emitError(getLocation()) << "Expected group name after `def`";
        return nullptr;
    }
    std::string GroupName = IdentifierStr;

    // Swallow the group name
    getNextToken();

    // This must be followed by a `(`:
    if (CurrentToken != '(') {
        mlir::emitError(getLocation()) << "Group parsing: expected ( after group name.";
        return nullptr;
    }

    // We swallow the `(` with the initial `getNextToken` in the loop below.
    // On successive iterations this will swallow the comma, if there is one.

    // This is followed by a comma separated list of generator names:
    std::vector<std::string> GroupGenerators;
    do {
        getNextToken(); // TODO: this can be moved into the `while`
        if (CurrentToken != Token::tok_identifier) {
            mlir::emitError(getLocation()) << "Group parsing: expected comma separated list of generators.";
            return nullptr;
        }
        GroupGenerators.push_back(IdentifierStr);
        getNextToken();

    } while(CurrentToken == ',');

    if (CurrentToken != ')') {
        mlir::emitError(getLocation()) << "Group parsing: expected ) after group argument list.";
        return nullptr;
    }

    getNextToken();

    return std::make_unique<GroupPrototypeAST>(GroupName, std::move(GroupGenerators), getLocation());
} 

std::unique_ptr<ExprAST> Parser::ParseGroup() {
    // Eat `def`.
    getNextToken(); 
    
    auto GroupPrototype = ParseGroupPrototype();
    if (!GroupPrototype) return nullptr;

    std::vector<std::unique_ptr<ExprAST>> Rules;

    if (CurrentToken != '{') return std::make_unique<GroupAST>(std::move(GroupPrototype), std::move(Rules), getLocation());

    do {
        getNextToken();
        auto Rule = ParseExpression();
        if (!Rule) return nullptr;
        
        Rules.push_back(std::move(Rule));
    } while (CurrentToken == ',');


    if (CurrentToken != '}') {
        mlir::emitError(mlir::UnknownLoc::get(&Context)) << "Group generators should end with '}'.";
        return nullptr;
    }
    getNextToken();
    return std::make_unique<GroupAST>(std::move(GroupPrototype), std::move(Rules), getLocation());
}

/// Constraints are of the form
///
/// `constr <expr> = <expr>`
///
/// With the expectation that both expressions belong to the
/// same group.
std::unique_ptr<ExprAST> Parser::ParseConstraint() {
    
    if (CurrentToken != Token::tok_constr) {
        // TODO: we need an unreachable error here.
    }

    getNextToken(); 

    std::unique_ptr<VariableExprAST> LHS = ParseIdentifierExpr();

    if (LHS == NULL) {
        mlir::emitError(getLocation()) << "Failed to parse LHS of constr.";
    }

    if (CurrentToken != '=') {
        mlir::emitError(getLocation()) << "Expected `=` in constr, found `" << CurrentToken << "`";
    }

    getNextToken(); 

    std::unique_ptr<VariableExprAST> RHS = ParseIdentifierExpr();

    if (RHS == NULL) {
        mlir::emitError(getLocation()) << "Failed to parse RHS of constr.";
    }

    return std::make_unique<ConstrAST>(std::move(LHS), std::move(RHS), getLocation());
}

std::unique_ptr<ExprAST> Parser::ParseAssign() {

    std::unique_ptr<VariableExprAST> LHS = ParseIdentifierExpr();

    if (CurrentToken != '=') {
        mlir::emitError(mlir::UnknownLoc::get(&Context)) << "Currently only group definitions and assignments are supported.";
        return nullptr;
    }

    getNextToken();

    auto RHS = ParseExpression();

    if (!RHS) {
        mlir::emitError(mlir::UnknownLoc::get(&Context)) << "Assignment RHS could not be parsed.";
        return nullptr;
    }

    return std::make_unique<AssignAST>(std::move(LHS), std::move(RHS), getLocation());
}

Program Parser::Parse() {

    std::vector<std::unique_ptr<ExprAST>> RootNodeVector;

    // Pre-load the first token.
    getNextToken();

    while(true) {
        switch (CurrentToken) {
            case Token::tok_eof:
                return std::move(RootNodeVector);
            case ';':
                getNextToken();
                break;
            case Token::tok_def: {
                auto Definition = ParseGroup();
                if (Definition != NULL) {
                    RootNodeVector.push_back(std::move(Definition));
                }
                break;
            }
            case Token::tok_constr: {
                auto Constraint = ParseConstraint();
                if (Constraint != NULL) {
                    RootNodeVector.push_back(std::move(Constraint));
                }
                break;
            }
            default: {
                auto Assign = ParseAssign();
                if (Assign != NULL) {
                    RootNodeVector.push_back(std::move(Assign));
                }
            }
        }
    } 

    return Program(std::move(RootNodeVector));
}

#endif // FRONTEND_PARSER_CPP
