The goal of this project is to create a language, in which one can play with finitely generated abelian groups.

## Groups

### Defining a group

A group is defined by the `def` keyword:

```
    # defining an trivial group.
    def G();
```

There are two sets of arguments one can provide:
- in `()` one specifies the generators of the group,
- in `{}` one specifies relations between the generators of the group.

Thus, one can define $\Z$ by setting:
```
    def Z(g){};
```

Or, one can define $\Z/2$ by writing:
```
    def Z2(g){g + g = e};
```

**Note.** Every group should automatically has an element $e$.

One can also take inverses of elements, simply by using `-` instead:
```
    def G(g);

    # inverse of $g$
    h = G(-g);
```

### Operations in groups

Obtaining elements of the group are performed as follows:

```
    def G(g);

    h = g + g;
    # or:
    G: h = g + g;
```
The first option does not specify the group we are working in. This is convenient when there is only one group in question, or if the groups in question use distinct labels for their elements. The second option is to explicitly specify the group we are working in by starting the line by `G:`.

### Ideas / Plans

1. **Binary op support.**  
    The AST implemented for AlgLang has a `BinaryExprAST` class, which might sound overkill, since we should be fine with only supporting `+` and `-` operations. However, it might be a good idea to include the option to define custom operations, like conjugation:
    ```
        # Concept of defining a custom binary operation
        bin_op | [g, h] = h - g

        G = def[k, l]{};

        t_1 = k | l;
        t_2 = l - k;

        # These two are the same!
    ```
2. **Group homomorphisms.**
    Since currently, and likely until any big upgrade, the language supports only abelian groups -- or _typed vectors_ -- it might make sense to define group homomorphisms, subgroups, cosets, etc. 


