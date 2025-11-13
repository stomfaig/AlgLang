#ifndef FRONTEND_PARSER_CPP
#define FRONTEND_PARSER_CPP

#include <string>
#include <iostream>
#include <map>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "ast.h"
#include "parser.h"
#include "mlirgen.h"

static std::string IdentifierStr;
static double NumVal;
static std::map<char, int> BinopPrecedence;
static int CurrentToken;

static int GetTokPrecedence() {
  if (!isascii(CurrentToken))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurrentToken];
  if (TokPrec <= 0) return -1;
  return TokPrec;
}

static int gettok() {
    static int LastChar = ' ';

    // Swallow spaces
    while (isspace(LastChar)) {
        LastChar = getchar();
    }

    // First, we check if the string starts with a number or a letter:
    // If it is a letter, it must be a command or an identifier.
    if (isalpha(LastChar)) { // The word must start with a letter
        IdentifierStr = LastChar;

        // Read all letters _and_ numbers
        while (isalnum((LastChar = getchar())))
            IdentifierStr += LastChar;

        // Check if we have read a command
        if (IdentifierStr == "def")
            return Token::tok_def;
        else
            return Token::tok_identifier;
    }

    // If it is a number, it must be a number
    // In our case only integers make sense.
    if (isdigit(LastChar)) {
        std::string NumString;
        NumString = LastChar;

        while(isdigit((LastChar = getchar()))) 
            NumString += LastChar;

        NumVal = strtod(NumString.c_str(), 0);
        return Token::tok_number;
    }

    // Finally, we handle comments:
    if (LastChar == '#') {
        // Swallow all characters until the end of the line or file
        while (LastChar != EOF || LastChar != '\n' || LastChar != '\r')
            LastChar = getchar();

        // If it is not the end of the file, we set LastChar to the next char,
        // and call `gettok` to return the next token.

        // NB. The tutorial does not include the `.. = getchar();` call, which seems a bit silly.
        if (LastChar != EOF)
            LastChar = getchar();
        
        return gettok();
    }

    // If none of the above matched, we either found EOF or a random char.
    // EOF is easy:
    if (LastChar == EOF) return Token::tok_eof;

    // Otherwise, we update LastChar, and return the ASCII code of the
    // unrecognised char:
    int RetChar = LastChar;
    LastChar = getchar();
    return RetChar;
}

static int getNextToken() {
    CurrentToken = gettok();

    return CurrentToken;
}

// This method assumes that the next token is a `(`
static std::unique_ptr<ExprAST> ParseParenExpr() {
    getNextToken(); // Eat `(`

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

static std::unique_ptr<VariableExprAST> ParseIdentifierExpr() {
    std::string IdentifierName = IdentifierStr;
    getNextToken();
    return std::make_unique<VariableExprAST>(IdentifierName);
}

// This method assumes that the next token is a `Token::tok_number`
static std::unique_ptr<ExprAST> ParseNumberExpr() {
    std::unique_ptr<ExprAST> result = std::make_unique<NumberExprAST>(NumVal);
    getNextToken();
    return result;
}

static std::unique_ptr<ExprAST> ParsePrimary() {
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

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExpressionPrecedence, std::unique_ptr<ExprAST> LHS) {

    while (true) {
        int TokenPrecedence = GetTokPrecedence();

        // This implicitly also checks for invalid operations.
        if (TokenPrecedence < ExpressionPrecedence) return LHS;

        int BinaryOperation = CurrentToken;
        getNextToken();

        std::unique_ptr<ExprAST> RHS = ParsePrimary();

        if (!RHS) {
            fprintf(stderr, "Parsing first RHS in ParseBinOpRHS failed on token %c", CurrentToken);
            return nullptr;
        }

        int NextPrecedence = GetTokPrecedence();

        if (TokenPrecedence < NextPrecedence) {
            RHS = ParseBinOpRHS(TokenPrecedence + 1, std::move(RHS));
            if (!RHS)
                return nullptr;
        }

        LHS = std::make_unique<BinaryOpAST>(BinaryOperation, std::move(LHS), std::move(RHS));
    }
}

static std::unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS) {
        fprintf(stderr, "LHS parsing failed in ParseExpression.");
        return nullptr;
    }

    return ParseBinOpRHS(0, std::move(LHS));
}

static std::unique_ptr<GroupPrototypeAST>ParseGroupPrototype() {

    // The parsing of the protype has the responsibility to read everything between the `def` keyword and the body of the definition.

    // This must be followed by the name of the group
    if (CurrentToken != Token::tok_identifier) {
        fprintf(stderr, "Error: Expected group name after `def`");
        return nullptr;
    }
    std::string GroupName = IdentifierStr;

    // Swallow the group name
    getNextToken();

    // This must be followed by a `(`:
    if (CurrentToken != '(') {
        fprintf(stderr, "Error: Group parsing: expected ( after group name.");
        return nullptr;
    }

    // We swallow the `(` with the initial `getNextToken` in the loop below.
    // On successive iterations this will swallow the comma, if there is one.

    // This is followed by a comma separated list of generator names:
    std::vector<std::string> GroupGenerators;
    do {
        getNextToken(); // TODO: this can be moved into the `while`
        if (CurrentToken != Token::tok_identifier) {
            fprintf(stderr, "Error: Group parsing: expected comma separated list of generators.");
            return nullptr;
        }
        GroupGenerators.push_back(IdentifierStr);
        getNextToken();

    } while(CurrentToken == ',');

    if (CurrentToken != ')') {
        fprintf(stderr, "Error: Group parsing: expected ) after group argument list.");
        return nullptr;
    }

    getNextToken();

    return std::make_unique<GroupPrototypeAST>(GroupName, std::move(GroupGenerators));
} 

static std::unique_ptr<ExprAST> ParseGroup() {
    // Eat `def`.
    getNextToken(); 
    
    auto GroupPrototype = ParseGroupPrototype();
    if (!GroupPrototype) return nullptr;

    std::vector<std::unique_ptr<ExprAST>> Rules;

    if (CurrentToken != '{') return std::make_unique<GroupAST>(std::move(GroupPrototype), std::move(Rules));

    do {
        getNextToken();
        auto Rule = ParseExpression();
        if (!Rule) {
            fprintf(stderr, "Rule expression parsing failed.");
            return nullptr;
        }
        
        Rules.push_back(std::move(Rule));
    } while (CurrentToken == ',');


    if (CurrentToken != '}') {
        fprintf(stderr, "Error: Group generators should end with '}'.");
        return nullptr;
    }
    getNextToken();
    return std::make_unique<GroupAST>(std::move(GroupPrototype), std::move(Rules));
}

static std::unique_ptr<ExprAST> ParseAssign() {

    std::unique_ptr<VariableExprAST> LHS = ParseIdentifierExpr();

    if (CurrentToken != '=') {
        fprintf(stderr, "Currently only group definitions and assignments are supported.");
        return nullptr;
    }

    getNextToken();

    auto RHS = ParseExpression();

    if (!RHS) {
        fprintf(stderr, "Assignment RHS could not be parsed.");
        return nullptr;
    }

    return std::make_unique<AssignAST>(std::move(LHS), std::move(RHS));
}

int run() {
    BinopPrecedence['+'] = 10;
    BinopPrecedence['-'] = 10;
    BinopPrecedence['*'] = 30;

    mlir::MLIRContext Context;
    Context.getOrLoadDialect<mlir::alg::AlgDialect>();
    Context.getOrLoadDialect<mlir::arith::ArithDialect>();

    MLIRGenImpl Gen(Context);

    fprintf(stderr, ">>> ");
    getNextToken();

    while(true) {
        switch (CurrentToken) {
            case Token::tok_eof:
                return 0;
            case ';': // ignore top-level semicolons.
                getNextToken();
                break;
            case Token::tok_def: {
                auto Definition = ParseGroup();
                if (Definition) {
                    Definition->dump();

                    // Call into MLIR code generation.
                    auto Module = Gen.mlirModuleGen(*Definition);
                    Module.print(llvm::outs());
                } else
                    std::cerr << "No AST produced!\n";
                break;
            }
            default: {
                auto Assign = ParseAssign();

                if (Assign) {
                    Assign->dump();

                    auto Module = Gen.mlirModuleGen(*Assign);
                    Module.print(llvm::outs());
                } else
                    std::cerr << "No AST produced!\n";
                break;
            }
        }
        fprintf(stderr, ">>> ");
    }
    

    return 0;
}

#endif // FRONTEND_PARSER_CPP