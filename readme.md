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

In the following, example, we are going to do the following:

1. Define a group on three generators,
2. Perform an addition of elements,
3. Perform a subtraction of elements,
4. Perform a scalar multiplication,
5. Lower to a mix of `arith` and `vector` dialects, essentially implementing all the previous operations via vectors.

```AlgLang
>>> def G(g,h);
...
>>> a = g + h;
...
>>> b = h - k;
...
>>> c = h * 5;
...
```

This is translated to the following Alg dialect based code:

```
module {
  # definition of the three elements as 3 basis vectors, with the extra group attribute.
  %0 = "alg.element.create"() <{data = [1 : i32, 0 : i32, 0 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<3xi32>>
  %1 = "alg.element.create"() <{data = [0 : i32, 1 : i32, 0 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<3xi32>>
  %2 = "alg.element.create"() <{data = [0 : i32, 0 : i32, 1 : i32], group = #alg.grp<"G">}> : () -> !alg.elem<<"G"> : vector<3xi32>>
  # adding two elements via the "alg.add" operation.
  %3 = "alg.add"(%0, %1) : (!alg.elem<<"G"> : vector<3xi32>>, !alg.elem<<"G"> : vector<3xi32>>) -> !alg.elem<<"G"> : vector<3xi32>>
  # subtracting two elements, by 1. taking the unary negation of the second, and then adding the resulting elements.
  %4 = "alg.un_neg"(%2) : (!alg.elem<<"G"> : vector<3xi32>>) -> !alg.elem<<"G"> : vector<3xi32>>
  %5 = "alg.add"(%1, %4) : (!alg.elem<<"G"> : vector<3xi32>>, !alg.elem<<"G"> : vector<3xi32>>) -> !alg.elem<<"G"> : vector<3xi32>>
  # scalar multiplying the an element by a constant via the "alg.sc_mul" operation.
  %c5_i32 = arith.constant 5 : i32
  %6 = "alg.sc_mul"(%1, %c5_i32) : (!alg.elem<<"G"> : vector<3xi32>>, i32) -> !alg.elem<<"G"> : vector<3xi32>>
}
```

Finally, this is lowered to a mix of the builtin `arith` and `vector` dialects:

```
module {
  # Define the 3 elements via the builtin vector type, forgetting the additional group structure.
  # -- no simplification passes are applied currently.
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0_i32_0 = arith.constant 0 : i32
  %0 = vector.from_elements %c1_i32, %c0_i32, %c0_i32_0 : vector<3xi32>
  %c0_i32_1 = arith.constant 0 : i32
  %c1_i32_2 = arith.constant 1 : i32
  %c0_i32_3 = arith.constant 0 : i32
  %1 = vector.from_elements %c0_i32_1, %c1_i32_2, %c0_i32_3 : vector<3xi32>
  %c0_i32_4 = arith.constant 0 : i32
  %c0_i32_5 = arith.constant 0 : i32
  %c1_i32_6 = arith.constant 1 : i32
  %2 = vector.from_elements %c0_i32_4, %c0_i32_5, %c1_i32_6 : vector<3xi32>
  # addition of elements translates to addition of vectors.
  %3 = arith.addi %0, %1 : vector<3xi32>
  # Unary negation is translated into multiplying by `-1`
  # Internally scalar multiplication is lowered to broadcasting a constant,
  # and applying element-wise integer multiplication
  %c-1_i32 = arith.constant -1 : i32
  %4 = vector.broadcast %c-1_i32 : i32 to vector<3xi32>
  %5 = arith.muli %4, %2 : vector<3xi32>
  # the addition half of the subtraction
  %6 = arith.addi %1, %5 : vector<3xi32>
  # final multiplication by constant
  %c5_i32 = arith.constant 5 : i32
  %7 = vector.broadcast %c5_i32 : i32 to vector<3xi32>
  %8 = arith.muli %7, %1 : vector<3xi32>
}
```