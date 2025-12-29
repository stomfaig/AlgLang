#include "constraints.h"
#include "llvm/Support/ErrorHandling.h"
#include <iostream>
#include <ostream>
#include <type_traits>

void ConstraintTable::introduceConstraint(const AlgVec& constraint_vec) {

    if (this->n != constraint_vec.size()) {
        /// TODO: report error
    }

    AlgVec reduced_constraint_vec = reduceElement(constraint_vec);
    
    size_t max_idx = n-1;
    for (; max_idx >= 0; max_idx--) {
        if (reduced_constraint_vec[max_idx] != 0)
            break;
    }

    if (max_idx < 0) return;

    if (reduced_constraint_vec[max_idx] < 0)
        reduced_constraint_vec *= -1;

    int NumToReduce = reduced_constraint_vec[max_idx];
    reduced_constraint_vec[max_idx] = 0;

    Constraints.push_back(Constraint {
        max_idx,
        NumToReduce,
        reduced_constraint_vec
    });

}

AlgVec ConstraintTable::reduceElement(const AlgVec& constraint) {

    if (this->n != constraint.size()) {
        /// TODO: report error.
    }

    AlgVec internal_constraint = constraint;

    for (Constraint c : this->Constraints) {
        while (internal_constraint[c.ReducedIndex] > c.NumToReduce) {
            internal_constraint[c.ReducedIndex] -= c.NumToReduce;
            internal_constraint += c.Result;
        }
        while (internal_constraint[c.ReducedIndex] < 0) {
            internal_constraint[c.ReducedIndex] += c.NumToReduce;
            internal_constraint -= c.Result;
        }
    }

    return internal_constraint;

}

void AlgVec::dump() {
    size_t vec_size = size();
    std::cout << "Vector(" << vec_size << "), ";
    for (size_t i = 0; i < vec_size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void Constraint::dump() {
    std::cout << "Constraint:\n\tRI: " << ReducedIndex << "\n\tNTR: " << NumToReduce << "\n\t";
    Result.dump();
}

void ConstraintTable::dump() {
    std::cout << "ConstraintTable state:" << std::endl;
    for (auto constraint : Constraints) {
        constraint.dump();
    }
}