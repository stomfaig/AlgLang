#ifndef FRONTEND_AST_GROUP_RESOLUTION_PASS_H
#define FRONTEND_AST_GROUP_RESOLUTION_PASS_H

#include "ast.h"
#include <map>
#include <set>
#include <string>
#include <vector>

/// AST pass for resolving the exact group type associated with
/// each invoked variable. 
/// We do this, by keeping a record of all the possible groups
/// that a variable name has been associated with. Then, given
/// each top level statement, we collect all the variables that
/// are involved in the statement, and take the intersection of
/// all of their sets of groups with which they are associated.
/// If this intersection is a singleton, we found the group, if
/// it is anything else, we know that the input code was wrong.
class GroupResolutionPass : public ASTVisitor {
private:
    std::set<std::string> AllGroups;        // Used to store all the available groups.
    std::set<std::string> CompatibleGroups;     // The set of compatible groups for the current assigns
    std::vector<const VariableExprAST*> CurrentAssigns;     // Variables for which the group will be determined
    std::map<std::string, std::set<std::string>> VarCompatibleGroups;   // The set of all groups found to be compatible with each var so far
    std::map<const VariableExprAST*, std::string> VariableGroups;       // Final table for all the resolved variables

    /// The following two functions are used to setup and setdown
    /// the state of the Pass in top-level nodes.
    /// This function:
    /// 1. empties CurrentAssigns, 2. empties CompatibleGroups
    void topLevelSetup();
    /// This function, for each node in CurrentAssigns, marks the
    /// node with the inferred group, or reports failure. 
    void topLevelSetdown();
    /// Method for introducing a new group that a given
    /// name is compatible with.
    void addCompatibleGroup(std::string varname, std::string groupname);

public:
    void visit(const VariableExprAST&) override;
    void visit(const AssignAST&) override;
    void visit(const BinaryOpAST&) override;
    void visit(const GroupAST&) override;
    void visit(const GroupPrototypeAST&) override;
    void visit(const ConstrAST&) override;

    void run(Program &program);
};

#endif // FRONTEND_AST_GROUP_RESOLUTION_PASS_H