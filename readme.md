## `AlgLang` as a simple MLIR example

This repo contains a demo project for using MLIR. `ALgLang` is a language, where one can construct and manipulate [Abelian groups](https://en.wikipedia.org/wiki/Abelian_group). This domain is chosen as a subsistitute of the matrix manipulation language used in [Toy](https://mlir.llvm.org/docs/Tutorials/Toy/), to give one a playground, where all the ideas of simplyfying AST's, designing passes has to be done from scratch.


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

