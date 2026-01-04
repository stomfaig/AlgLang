#include <cstddef>
#include <memory>
#include <vector>


class AlgVec {
    std::vector<int> data;

public:
    explicit AlgVec(size_t n) : data(n) {}

    size_t size() const { return data.size(); }
    bool is_zero();

    AlgVec& operator+=(const AlgVec& rhs);
    AlgVec operator -=(const AlgVec& rhs);
    AlgVec& operator*=(const int rhs);
    bool operator==(const AlgVec& rhs);
    bool operator!=(const AlgVec& rhs) {
        return !(*this == rhs);
    }
    int& operator[](size_t i) { return data[i]; }
    const int& operator[](size_t i) const { return data[i]; }

    void dump();
};


/// Constraints are used to keep track of constrained introduced in- and
/// outside of class definitions.
/// They store the index that they are reducing, and also the number of components
/// of that index they need to reduce. They also store what the element reduces to.
///
/// TODO: maybe add an example.
struct Constraint {
    size_t ReducedIndex;
    int NumToReduce;
    AlgVec Result;

    static Constraint from_vec(AlgVec);

    void dump();
};

/// We use constraint tables to keep track of all the constraints that have been introduced in a group.
/// Each entry in the con
class ConstraintTable {
    size_t n;
    std::vector<Constraint> Constraints;
    
public:
    ConstraintTable (size_t n) : n(n) {}

    /// Given a vector, that should be considered 0, this method
    /// (a) simplifies the vector using existing constrains,
    /// (b) if the simplified expressions is not trivial, adds it to `Constraints`
    /// E.g., given `constr a = b;`, we call `introduceConstraint(a-b)`.
    /// TODO: maybe return a success / failure value
    void introduceConstraint(AlgVec);
    /// Given an element (in vector form), this method returns a 
    /// simplified version of the element using the constraints
    /// stored in this objects.
    AlgVec reduceElement(AlgVec);

    /// This method internally simplifies all the stored constraints
    /// to their simplest form
    void simplify();

    void dump();
};