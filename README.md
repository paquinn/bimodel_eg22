# BiModel

## Instalation and Setup

## Language

### Parameters
Any lowercase value or tuple denotes a parameter that can be optimized for.

Ex, `s = 3.0` or `x, y, z = 0.0, 0.0, 0.0`.

Parameters can also refer to expressions of previous parameters.

Ex, `y = m * sin(pi * (s + 1))`

### Operations
Operations take arguments as keywords.

Ex, `Cube(size=s, location=(x, y, z))`.

These arguments can also be expressions.

Ex, `Extrude(length=3. * s, select=...)`.

If an arguments is given as a constant, it is not optimized for.

Ex, `Cube(size=3., location=(x, y, 0.0))`.

Arguments can be omitted in which case the defaults shown bellow will be used.

### Named verts

All operations that create geometry can be assigned an uppercase name.

Ex, `B1 = Box(size=1.0, location=(0.0, 0.0, 0.0))`.

These names can then be used to select which vertices the operation will be based on.

Ex, `E1 = Extrude(length=3.0, select=['v0@B1', 'v1@B1', 'v2@B1', 'v3@B1'])`.

All operations that create geometry leave the new vertices selected for each of use. This allows for easily chaining operations.

```
Extrude(length=1.0, select=['f0@E1'])
Resize(value=(2.0, 2.0, 2.0))
Extrude(length=2.0)
Resize(value=(3.0, 3.0, 3.0))
...
```


### Mesh Primitives

* `Cube`
    - `location = (0.0, 0.0, 0.0)` 
    - `size = 1.0`
* `Grid`
    - `location = (0.0, 0.0, 0.0)` 
    - `size = 2.0`
    - `subs = (2, 2)`
        - Constant value.

### Mesh Operations

* `Extrude`
    - `length = 1.0`
    - `select`
        - Can be one or more faces. Adjacent faces groups with the same normal will be connected.
