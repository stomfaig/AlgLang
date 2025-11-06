The goal of this project is to create a language, in which one can play with finitely generated abelian groups.

## Groups

### Defining a group

A group is defined by the `def` keyword:

```
    # defining an trivial group.
    G = def[]{};
```

There are two sets of arguments one can provide:
- in `[]` one specifies the generators of the group,
- in `{}` one specifies relations between the generators of the group.

Thus, one can define $\Z$ by setting:
```
    Z = def[g]{};
```

Or, one can define $\Z/2$ by writing:
```
    Z2 = def[g]{g + g = e};
```

**Note.** Every group automatically has an element $e$.

One can also take inverses of elements, simply by using `-` instead:
```
    G = def[g];

    # inverse of $g$
    h = G(-g);
```

### Operations in groups

Obtaining elements of the group are performed as follows:

```
    G = def[g];

    h = G(g + g);
```

That is, we write the operation we would like to perform in `()`'s after the name of the group. This can also be done without specifying the group in which operations to be performed, by simply writing
```
    h = g + g;
```
in which case the group to which the item belongs is identified automatically.


### TODO: write these into proper docs.

1. **Binary op supports.**  
    The AST implemented for AlgLang has a `BinaryExprAST` class, which might sound overkill, since we should be fine with only supporting `+` and `-` operations. However, it might be a good idea to include the option to define custom operations, like conjugation:
    ```
        # Concept of defining a custom binary operation
        bin_op | [g, h] = h - g

        G = def[k, l]{};

        t_1 = k | l;
        t_2 = l - k;

        # These two are the same!

        # NB. the exact syntax here might change.
    ```

## Implementation


### Parsing.



