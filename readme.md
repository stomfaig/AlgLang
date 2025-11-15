## `AlgLang` as a simple MLIR example

This repo contains a demo project for using MLIR. `ALgLang` is a language, where one can construct and manipulate [Abelian groups](https://en.wikipedia.org/wiki/Abelian_group). This domain is chosen as a substitute of the matrix manipulation language used in [Toy](https://mlir.llvm.org/docs/Tutorials/Toy/), to give one a playground, where all the ideas of simplifying AST's, designing passes has to be done from scratch.  

While at first it might seem that Abelian groups only of interest to mathematicians and they are completely irrelevant for any computational applications, I am keen to highlight that the structure of Abelian groups is very similar to that of "glued vector lattices." The following brief discussion justifies this.

1. Given generators $g_1, \ldots, g_n$ of an Abelian group, we can represent the elements of the group as vectors in $\Z^n$. Addition and subtraction of elements maps exactly to addition and subtraction of their representing vectors.
2. Since each element belongs to a group, there is extra type-checking to be done compared to simple vector manipulation: only elements that belong to the same group can be added, even if their representation vectors otherwise have compatible sizes. Note also, that this is purely compile time information, so after typechecking `AlgLang` can be lowered to a vector manipulation dialect, and thus to machine code.
3. The language also exposes another difficulty, where some "vectors" can have multiple representations. For example, consider the group, defined in `AlgLang` as:
  ```AlgLang
    def G(g, h) {
      g - 2 * h
    };
  ```
  The representations of the elements of this language are then linear combinations of the vectors $v_g = (1, 0)$ and $v_h = (0, 1)$, with the extra condition, that $(1, 0) = (0, 2)$.


### How does `AlgLang` work?

In `AlgLang` one can define groups using the `def` keyword, by specifying the name of the group, its generators, and the relations that hold between the generators.

We use convention that the group operation of every group is denoted by `+` and `n * g` is equivalent to writing `g + g + ... + g`, where `g` appears `n` times. Similarly, the inverse of an element is denoted as `-n`.

For example, one could define the integers modulo 2 as follows:

```AlgLang
def G(g) {
    2 * g
};
```

Currently, the only other supported operation in `AlgLang`'s parser is untyped assignment:

```AlgLang
# Free group with generator `g`
def G(g);

k = 4 * g;
```

Here, the fact that `k` is also an element of the group `G` is automatically inferred, since `g` is a member of `G`. 

### Quickstart

The easiest way to get started is using CMake and Ninja. Once both are installed, run the following in the current directory:

```bash
mkdir build && cd build

cmake .. -DMLIR_DIR "<YOUR-LLVM-INSTALLATION>/llvm-install/lib/cmake/mlir" \
-DLLVM_DIR "<llvm-project-repo>" \
-DFRONTEND "<ON/OFF>"

ninja
```

The frontend flag controls whether the build file created will also compile the interpreter. Without it only the MLIR dialect is compiled.

### Example

The following example illustrates the current state of the language. The lines starting with `#` are comments added afterwards. Note that since `AlgLang` currently dumps all the code into a single module, since there is no logical flow or functions available.

```
# define a group with two generators `g` and `h`
>>> def G(g,h);
# print the produced AST node:
GroupAST
  GroupPrototype(G) [g, h]
# print the code in the module:
module {
  # define two SSA's with type `alg.elem`, with a given representation vector, and alg.grp<"G"> attribute, to mark their group membership.
  %0 = "alg.element.create"() <{data = [1 : i32, 0 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
  %1 = "alg.element.create"() <{data = [0 : i32, 1 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
}
# we now declare a new variable `t` as the sum of `g` and `h`.
>>> t = g + h;
# produced AST node:
Assign
  VariableExpr(t)
  BinaryOp(+)
    VariableExpr(g)
    VariableExpr(h)
module {
  %0 = "alg.element.create"() <{data = [1 : i32, 0 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
  %1 = "alg.element.create"() <{data = [0 : i32, 1 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
  # we add a third SSA assignment: and `alg.add` op, taking the two previously defined ops, and returning another element of the same group.
  %2 = "alg.add"(%0, %1) : (!alg.elem<<"G"> : vector<2xi64>>, !alg.elem<<"G"> : vector<2xi64>>) -> !alg.elem<<"G"> : vector<2xi64>>
}
# finally, we try out the `sc_mul`, or "scalar multiplication" op. This is meant to be a shorthand for `t + t + t + t` in this case:
>>> l = t * 4;
# AST node:
Assign
  VariableExpr(l)
  BinaryOp(*)
    VariableExpr(t)
    NumberExpr(4)
module {
  %0 = "alg.element.create"() <{data = [1 : i32, 0 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
  %1 = "alg.element.create"() <{data = [0 : i32, 1 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<2xi64>>
  %2 = "alg.add"(%0, %1) : (!alg.elem<<"G"> : vector<2xi64>>, !alg.elem<<"G"> : vector<2xi64>>) -> !alg.elem<<"G"> : vector<2xi64>>
  # we define an `arith.constant`, this is emitted by the processor of `NumberExpr(4)`
  %c4_i32 = arith.constant 4 : i32
  # we define a new SSA, based on the `alg.sc_mul` op, with the correct arguments.
  %3 = "alg.sc_mul"(%2, %c4_i32) : (!alg.elem<<"G"> : vector<2xi64>>, i32) -> !alg.elem<<"G"> : vector<2xi64>>
}

```