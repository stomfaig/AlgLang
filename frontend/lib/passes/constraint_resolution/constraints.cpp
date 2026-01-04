#include "constraints.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

AlgVec& AlgVec::operator+=(const AlgVec& rhs) {
    if (size() != rhs.size()) {
        // TODO: report error
    }

    for (size_t i = 0; i < size(); i++) {
        data[i] += rhs[i];
    }

    return *this;
}

AlgVec AlgVec::operator-=(const AlgVec& rhs) {
    if (size() != rhs.size()) {
        // TODO: report error
    }

    for (size_t i = 0; i < size(); i++) {
        data[i] -= rhs[i];
    }

    return *this;
}

AlgVec& AlgVec::operator*=(const int rhs) {
    for (size_t i = 0; i < size(); i++) {
        data[i] *= rhs;
    }

    return *this;
}

bool AlgVec::operator==(const AlgVec& rhs) {
    if (size() != rhs.size()) {
        /// TODO: report error
    }

    for (size_t i = 0; i < size(); i++) {
        if (data[i] != rhs[i])
            return false;
    }

    return true;
}

bool AlgVec::is_zero() {
    for(size_t i = 0; i < size(); i++) {
        if (data[i] != 0) return false;
    }

    return true;
}

void ConstraintTable::introduceConstraint(AlgVec constraint_vec) {

    if (this->n != constraint_vec.size()) {
        /// TODO: report error
    }
    
    auto constraint = Constraint::from_vec(std::move(constraint_vec));
    Constraints.push_back(std::move(constraint));

    simplify();
}

void ConstraintTable::simplify() {

    bool changed = false;

    do {
        changed = false;

        size_t i = 0;
        size_t constraint_num = Constraints.size();
        while (i < constraint_num) {
            auto constraint = Constraints.at(0);
            Constraints.erase(Constraints.begin());

            auto reduced = reduceElement(constraint.Result);

            if (reduced != constraint.Result) {
                changed = true;

                if (reduced.is_zero())
                    continue;

                auto new_constraint = Constraint::from_vec(reduced);
                Constraints.push_back(new_constraint);
                continue;
            } 

            Constraints.push_back(constraint);

            ++i;
        }
    } while (changed);
    
}

AlgVec ConstraintTable::reduceElement(AlgVec constraint) {

    if (this->n != constraint.size()) {
        /// TODO: report error.
    }

    for (auto &c : this->Constraints) {
        while (constraint[c.ReducedIndex] >= c.NumToReduce) {
            constraint -= c.Result;
        }
        while (constraint[c.ReducedIndex] < 0) {
            constraint += c.Result;
        }
    }

    return constraint;
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

Constraint Constraint::from_vec(AlgVec vec) {
    size_t max_idx = vec.size();
    
    while (max_idx > 0) {
        if (vec[--max_idx] != 0)
            break;
    }

    if (vec[max_idx] < 0)
        vec *= -1;

    int NumToReduce = vec[max_idx];

    return Constraint {
        max_idx,
        NumToReduce,
        std::move(vec)
    };
}

void ConstraintTable::dump() {
    std::cout << "ConstraintTable state:" << std::endl;
    for (auto &constraint : Constraints) {
        constraint.dump();
    }
}