# CAD Language Reference

## Selection:

A blender operation requires vertices (or other elements) to be selected in the active document, which serve as the geometric arguments to the function call.

In our language, a `Selection` can be defined by two constructs:

- `B.all()` selects everything created by op B
- `B.v([a,b,c])` or which will select the vertices at indexes a,b,c created by op B.

The blender API is used to mutate the document's current selection to correspond to this expression, and the subsequent operation using the selection is peformed. _Every non-primitive operation in the language_ takes an optional select parameter, which will define the selection. If this is not provided, the currently-active selection will be used.

## Language Operations

The modifier `+` is used after numeric types to denote that the input given is constrained to be positive.

### Input Types

```
type Axis = "X" | "Y" | "Z"
type RotationMode = "NORMAL" | "GLOBAL"
type Vec3 = (Float,Float,Float)
```

### Primitives

- `Cube(size: Float+, location: Vec3)`
  Creates a cube with side-length size with its center at `location`.

- `Grid(size: Float+, location: Vec3, subs: Vec2 = (2,2))`
  Creates a square grid of vertices in a single surface, with an overall edge length of `size`, starting at `location`, extending `subs[0]` and `subs[1]` elements in the `X` and `Y` directions respectively. The X-direction offsets between each element are computed as `size / subs[0]`, and respectively for Y.

### Transformations

- `Resize(value: Vec3)`
  Non-uniformly scales the selected vertices around the centroid, according to `value` in each respective dimension.

- `Translate(value: Vec3)`
  Translates the selected vertices by the given vector.

- `Rotate(value: Float, axis: Axis = "Z", mode: RotationMode = "NORMAL")`
  Rotates the selected vertices around the given axis with angle given by value in radians. Currently the only implemented mode is `Global` around the `X` and `Y` axes.

  Todos:

  - Rotate around any specified axis (perhaps by 2 points)
  - Implement `Normal` rotation mode.

### Topology modifications

- `Extrude(length: Float, dynamic: Bool = false)`
  Extrudes the selected vertices length. Currently works around the normal vector of the face the vertices end up selecting.

  Todos:

  - Extrude in an arbitrary direction
  - Extrude downwards, insetting a face instead of extruding it.

- `Bridge()`:
  Creates a bridge edge loop between the currently selected vertices. Expects the selected vertices to form two distinct, complete edge loops, which are then bridged between.

### Other

- `Select(select: Selection)`
  Manually mutates the current document selection as specified.
