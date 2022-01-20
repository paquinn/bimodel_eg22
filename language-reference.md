# CAD Language Reference

We give a full description of our language from the API perspective, illustrating the relevant concepts needed to both use and re-implement this language.

**Parameters**

Any lowercase numeric literal assigned to a variable or tuple or variables denotes a parameter that can be optimized for. For example:

```
s = 3.0
x, y, z = 0.0, 0.0, 0.0
```

No variables in our language can be reassigned to, and parameters cannot be created in loops. Variables can also be expressions of previous parameters, but these are not treated as parameters for the purposes of optimization.

    `y = m*sin(pi*(s+1))`

Notably, numeric literals in such expressions are treated as constants, not as parameters. As a result, we admit:

```
	C = fix(2)
```

which allows the definition of constant variables. Throughout the remainder of this document we will use the type <code><em>num </em></code>to refer to expressions that depend on parameters, and <code><em>const</em></code> to refer to expressions that are constant. We will use <em>num3 </em>to mean a tuple (<em>num,num,num). </em>All arguments have default values provided by our system if omitted. To use any variable in a <code><em>const</em> </code>expression, we must call the <code><em>const() </em></code>function to perform the cast, which will fail if the variable’s value depends on parameters.

We define the following mathematical operations available on parameters and constants:

- Arithmetic: (`+, -, *, /`)
- Exponentiation: (pow(), exp(), ln(), sqrt()). Note that the`a ** b`infix operator is a shortcut for `pow(a,b).`
- Trigonometry: `sin(x), cos(x)`

## Operations

### Mesh Primitives

```
Cube(size : num, location : num3)
```

    Creates a cube centered at <code><em>location </em></code>with edge length <code><em>size. </em></code>


    Creates the constraint `size > 0.`

```
Box(base : num3, dims : num3)
```

    Creates a `dims[0]xdims[1]xdims[2]`  box with the base point <code><em>base</em></code>.

```
Grid(size : num, location : num3, subs : const)
```

    Create a 2-dimensional plane with <code><em>subs<strong> </strong></em></code>subdivisions and edge length <code><em>size</em></code> centered at <code><em>location</em></code>.


    Creates the constraint `size > 0`.

```
Circle(radius : num, location : num3, fill : FILL, verts : const)
```

    Creates a circle with radius <code><em>radius</em> </code>and <code><em>verts<strong> </strong></em></code>vertices that define it, centered at <code><em>location</em></code>.  <code><em>Fill </em></code>is a string parameter which<em> </em>determines how the circle is filled topologically and can take one of the following values:

- `TRIFAN`: fills the circle with triangular faces which share a vertex in the middle.
- `NGON`: fills the circle with a single N-gon.
- `NOTHING`: does not fill; creates only the outer ring of vertices.

Creates the constraint `radius > 0`.

```
Cylinder(radius : num, depth : num, location : num3,
   fill : FILL, verts : const)
```

    Creates a cylinder with radius <code><em>radius</em></code> and depth <code><em>depth</em></code> centered at <code><em>location. </em></code>Similarly to <code>Circle</code> above, <code><em>verts<strong> </strong></em></code>determines the number of vertices that define the bases of the cylinder,<em> and<code> fill</code></em> determines the topology of the fill.


    Creates the constraints `radius > 0 `and` depth > 0.`

```
Sphere(radius : num, location : num3,
     nsegments : const, nrings : const)
```

    Creates a sphere of radius <code><em>radius</em></code> centered at <code><em>location</em></code>, where <code><em>nsegments</em></code> is the number of vertical segments and <code><em>nrings</em></code> is the number of horizontal segments.


    Creates the constraint `radius > 0`.

All parameters shown in bold affect mesh topology and are therefore required to be constant by our interpreter.

### Selection, Named Vertices and Referencing

Each operation mutates the “currently selected” set of vertices. This is a piece of state maintained by the underlying CAD system, and required in order to pass arguments to its API.

Primitives, when created, replace the current selection with the created primitive, allowing for chaining operations.

All operations that create geometry (Primitives and Geometry Modifiers) can be assigned to an uppercase variable. For example,

`B1 = Box(size=1.0, location=(0.0, 0.0, 0.0))`.

These names can then be used to:

- Select vertices created by the operation
- Reference the coordinates of vertices created by the operation as numeric values.

This is done by a reference expression. For example, the following selects the first and second vertices created by the above Box operation. Each vertex is always returned in the same order, so this selection is stable. To mutate the selection, the expression must be passed to the select keyword argument of any operation that supports it. By default, this is the `Select` operation.

```
  Select(select=B1.v([0,1]))
```

To reference particular coordinates, we can assign (or use, where applicable):

```
   a = B1.x(0) # The x coordinate of the first vertex created by B1
```

or` `

```
   v = B1.ref(4)
   a = v.z     # The z coordinate of the 4th vertex created by B1
```

An edge or face cannot be directly selected from our system. Instead, an edge or face is considered selected if all of its constituent vertices are selected.

### Transformations

If select is not provided, by default all transformations will use the active selection created by a previous transformation or selection command. Transformations do not create vertices and so cannot be assigned to a variable.

```
Translate(value : num3, select : Select)
```

    Translates the elements listed in `select` by the vector `value.`

```
Rotate(theta : num, axis: Axis, center: num3, select: Select)
```

    Rotates the vertices listed in <code><em>select</em></code> by <code><em>theta</em> </code>around the axis <code><em>axis</em>, </code>centered around point <code><em>center</em></code>. <code><em>Axis</em> </code>is a string of either<code> 'X', 'Y', </code>or<code> 'Z'. </code>

```
Resize(value : num3, select : Select)
```

    Scales the vertices in <code><em>select</em> </code>by each coordinate of <code><em>value</em> </code>in its respective axis


    Creates the constraints` value[0] > 0, value[1] > 0, value[2] > 0.`

### Geometry Modifiers

These operations can change the topology of existing geometry, and add new vertices. They can be assigned to a variable.

```
Chamfer(length : num, select : Select)
```

    Chamfers the edges listed in select. <code><em>length </em></code>is the distance between the newly created edges and the original edge (before running<em> <code>Chamfer</code></em>)<em>. </em>


    Creates constraints to ensure that <code><em>length </em></code>is smaller than the length of any edge truncated by the chamfer. This is done on the edges <em>at the time of the chamfer</em>, and does <em>not </em>constrain the final edge length after arbitrary transformations are applied.

```
Extrude(length : num, select : Select)
```

    Extrudes the faces listed in <code><em>select </em></code>by <code><em>length </em></code>along their corresponding normals. Creates a constraint<code> <em>length > 0.</em></code>

### Topology Modifiers

These operations do not affect the vertex positions in any way, they only adjust the topological state of the mesh. However, this affects objective functions defined on our program, as well as what selections count as faces or edges, and so must be executed.

```
DeleteVertex/DeleteEdge/DeleteFace(select)
```

    Deletes the vertices/edges/faces listed in <code><em>select</em>.</code>

```
Fill(select)
```

    Creates a face between the vertices in<code> <em>select.</em></code>

```
Flip(select)
```

    Flips the normal of each face in <code><em>select.</em></code>

```
Bridge(select)
```

    Creates a topological bridge between two strongly connected components of vertices of equal size in <code><em>select</em>.</code>This connects each vertex to a corresponding matching vertex with an edge, and creates the appropriate faces to retain a solid object.

### Selection

In addition to `Select`, we provide a few helpers for mutating the selection to a number of vertices that is not explicitly specified. This aids writing programs generically, for example we can select the top and bottom edge loops of a cylinder without hard-coding in the number of facets of the cylinder into the selection. These are:

```
Loop(select)
```

    Starting from an edge (2 vertices) in the current selection, selects a loop of edges that are connected in a line end-to-end passing through the selected edge.

```
Ring(select)
```

    Starting from an edge (2 vertices) in the current selection, selects a sequence of edges that are not connected, but on opposite sides to each other continuing along a face loop.

```
Path(select)
```

` `Selects the shortest (unweighted) path between a selection of two vertices.

```
Region(select)
 	Selects the vertices inside a loop of selected vertices in select.
Boundary(select)
```

` `Selects the vertices of the boundary of the vertices in <code><em>select</em></code>.

```
Connected(select)
```

Selects all vertices that are connected with an edge to a vertex in <code><em>select</em></code>.

### Constraints

```
Clamp(value : num, lower : num, upper : num)
	Creates user-defined constraints of  lower < value and value < upper.
```

One of either <code><em>upper</em></code> or <code><em>lower</em></code> may be omitted.

### Loops

We allow Python-style `for i in range() `loops, requiring the arguments to `range() `to be constant. Loops can be thought of as syntactic sugar for repeating a series of expressions in the body, and all loops are internally unrolled.
