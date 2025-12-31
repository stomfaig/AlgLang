#ifndef FRONTEND_AST_GROUP_RESOLUTION_PASS_CPP
#define FRONTEND_AST_GROUP_RESOLUTION_PASS_CPP

#include "passes/group_resolution_pass.h"
#include "ast.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>

void GroupResolutionPass::topLevelSetup() {
    CurrentAssigns.clear();
    CompatibleGroups = std::set<std::string>(AllGroups);
}

void GroupResolutionPass::topLevelSetdown() {
    // Figure out if there is a solution
    // If there is, mark it in the table
    if (CompatibleGroups.size() != 1) {
        std::cerr << "Cannot determine group of operation." << std::endl;
        return;
    }
    
    auto GroupName = *CompatibleGroups.begin();

    for (auto node : CurrentAssigns) {
        VariableGroups.insert({node, GroupName});
        addCompatibleGroup(node->getName(), GroupName);
    }

    /*std::cout << "Variable compatibility table" << std::endl;
    for (auto pair : VarCompatibleGroups) {
        std::cout << "Compatible groups with " << pair.first << ":\n\t";
        for (auto second : pair.second) {
            std::cout << second << " ";
        } 
        std::cout << std::endl;
    }

    std::cout << std::endl << "Assignment table:" << std::endl;
    for (auto pair : VariableGroups) {
        std::cout << pair.first->getName() << " : " << pair.second << std::endl;
    }

    std::cout << "Compatible groups:" << std::endl;
    for (auto gp : CompatibleGroups) {
        std::cout << gp << " ";
    }
    std::cout << std::endl;*/
}

void GroupResolutionPass::addCompatibleGroup(std::string varname, std::string groupname) {
    if (!VarCompatibleGroups.count(varname)) {
        VarCompatibleGroups.insert({varname, {}});
    }

    VarCompatibleGroups.at(varname).insert(groupname);
}

/// TODO: this function is not great
void GroupResolutionPass::visit(const VariableExprAST& node) {
    CurrentAssigns.push_back(&node);
    
    auto compatible_groups = VarCompatibleGroups.at(node.getName());

    std::set<std::string> result;

    std::set_intersection(
        CompatibleGroups.begin(),
        CompatibleGroups.end(),
        compatible_groups.begin(),
        compatible_groups.end(),
        std::inserter(result, result.begin())
    );

    CompatibleGroups = result;
}

void GroupResolutionPass::visit(const AssignAST& node) {
    topLevelSetup();

    node.getRHS().accept(*this);
    CurrentAssigns.push_back(&node.getLHS());

    topLevelSetdown();
}

void GroupResolutionPass::visit(const BinaryOpAST& node) {
    node.getLHS().accept(*this);
    node.getRHS().accept(*this);
}

void GroupResolutionPass::visit(const GroupAST& node) {
    // Here we should assign the current group to each parent

    node.getProto().accept(*this);
}

void GroupResolutionPass::visit(const GroupPrototypeAST& node) {
    AllGroups.insert(node.getName());
    for(auto generator : node.getGenerators()) {
        addCompatibleGroup(generator, node.getName());
    }
}

void GroupResolutionPass::visit(const ConstrAST& node) {
    topLevelSetup();

    // Get definition information from defining side
    
    topLevelSetdown();
}

void GroupResolutionPass::run(Program &program) {
    for (const auto &top_level_op : program.getTopLevelNodes()) {
        top_level_op->accept(*this);
    }
    program.setVariableGroups(VariableGroups);
}

#endif // FRONTEND_AST_GROUP_RESOLUTION_PASS_CPP