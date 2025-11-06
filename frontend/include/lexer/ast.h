#ifndef FRONTEND_ASH_H
#define FRONTEND_ASH_H

#include <string>
#include <iostream>

class ExprAST {
public:
    virtual ~ExprAST() = default;

    virtual void dump(std::ostream &os = std::cout, unsigned indent = 0) const;
};


class NumberExprAST : public ExprAST {
    // Has the extra property that it also stores a value.
    double Val;

public:
    NumberExprAST(double Val) : Val(Val) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
};


class VariableExprAST : public ExprAST {
    std::string Name;

public:
    VariableExprAST(const std::string &Name) : Name(Name) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
};


class BinaryOpAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;

public:
    BinaryOpAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS): Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
};

class GroupPrototypeAST : public ExprAST {
    std::string Name;
    std::vector<std::string> Generators;

public:
    GroupPrototypeAST(const std::string &Name, std::vector<std::string> Generators): Name(Name), Generators(std::move(Generators)) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
};

class GroupAST : public ExprAST {
    std::unique_ptr<GroupPrototypeAST> Proto;
    std::vector<std::unique_ptr<ExprAST>> Rules;

public:
    GroupAST(std::unique_ptr<GroupPrototypeAST> Proto, std::vector<std::unique_ptr<ExprAST>> Rules): Proto(std::move(Proto)), Rules(std::move(Rules)) {}

    void dump(std::ostream &os = std::cerr, unsigned indent = 0) const override;
};

#endif // FRONTEND_ASH_H