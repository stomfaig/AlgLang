#ifndef FRONTEND_ASH_H
#define FRONTEND_ASH_H

#include "mlir/IR/Location.h"
#include <string>
#include <iostream>

class ExprAST {
public:
    // LLVM-style RTTI discriminator
    // see: https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html
    enum ExprASTKind {
        EAK_Number,
        EAK_Variable,
        EAK_Assign,
        EAK_BinaryOp,
        EAK_GroupProto,
        EAK_Group,
    };

private:
    ExprASTKind Kind;
    mlir::Location Loc;

public:
    ExprAST(ExprASTKind Kind, mlir::Location Loc) : Kind(Kind), Loc(Loc) {}
    virtual ~ExprAST() = default;
    virtual void dump(std::ostream &os = std::cout, unsigned indent = 0) const;
    ExprASTKind getKind() const { return Kind; }
};


class NumberExprAST : public ExprAST {
    double Val;

public:
    NumberExprAST(double Val, mlir::Location Loc) : ExprAST(EAK_Number, Loc), Val(Val) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
    double getVal() const { return Val; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_Number;
    }
};


class VariableExprAST : public ExprAST {
    std::string Name;

public:
    VariableExprAST(const std::string &Name, mlir::Location Loc) : ExprAST(EAK_Variable, Loc), Name(Name) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
    const std::string &getName() const { return Name; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_Variable;
    }
};

class AssignAST : public ExprAST {
    std::unique_ptr<VariableExprAST> LHS;
    std::unique_ptr<ExprAST> RHS;

public:
    AssignAST(std::unique_ptr<VariableExprAST> LHS, std::unique_ptr<ExprAST> RHS, mlir::Location Loc): ExprAST(EAK_Assign, Loc), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
    const VariableExprAST &getLHS() const {return *LHS; }
    const ExprAST &getRHS() const {return *RHS; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_Assign;
    }
};


class BinaryOpAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;

public:
    BinaryOpAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS, mlir::Location Loc): ExprAST(EAK_BinaryOp, Loc), Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
    char getOp() const { return Op; }
    const ExprAST &getLHS() const { return *LHS; }
    const ExprAST &getRHS() const { return *RHS; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_BinaryOp;
    }
};

class GroupPrototypeAST : public ExprAST {
    std::string Name;
    std::vector<std::string> Generators;

public:
    GroupPrototypeAST(const std::string &Name, std::vector<std::string> Generators, mlir::Location Loc): ExprAST(EAK_GroupProto, Loc), Name(Name), Generators(std::move(Generators)) {}

    void dump(std::ostream &os = std::cout, unsigned indent = 0) const override;
    const std::string &getName() const { return Name; }
    const std::vector<std::string> &getGenerators() const { return Generators; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_GroupProto;
    }
};

class GroupAST : public ExprAST {
    std::unique_ptr<GroupPrototypeAST> Proto;
    std::vector<std::unique_ptr<ExprAST>> Rules;

public:
    GroupAST(std::unique_ptr<GroupPrototypeAST> Proto, std::vector<std::unique_ptr<ExprAST>> Rules, mlir::Location Loc): ExprAST(EAK_Group, Loc), Proto(std::move(Proto)), Rules(std::move(Rules)) {}

    void dump(std::ostream &os = std::cerr, unsigned indent = 0) const override;
    const GroupPrototypeAST &getProto() const { return *Proto; }
    const std::vector<std::unique_ptr<ExprAST>> &getRules() const { return Rules; }
    static bool classof(const ExprAST *Node) {
        return Node->getKind() == EAK_Group;
    }
};

#endif // FRONTEND_ASH_H