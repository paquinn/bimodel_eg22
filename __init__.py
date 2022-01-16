import bpy
import gpu
import bgl
from gpu_extras.batch import batch_for_shader

import ast
import astunparse
import sympy as sy
import bmesh
import mathutils
import pyperclip
import numpy as np
import scipy as sp
import re
import os
import igl
import sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from time import perf_counter_ns
import functools
import itertools

import cProfile
import pstats
import tempfile

from mathutils import Vector
from bpy.types import Operator, Panel
from scipy import optimize
from sympy import Array

from typing import Tuple, Dict, Union
import numbers

from . import dsl
from . import auto

diffVar = auto.diffVar
diffConst = auto.diffConst
valueOf = auto.valueOf
lift = auto.lift


def V(name, value):
  return diffVar(name, diffConst(value))


from . import crun

# Needed for evaluating some expressions
import math

# Rebind these expressions so we can use them interchangeably with diff types
# in the programs we evaluate.
def sin(x):  return x.sin()  if isinstance(x,auto.Diff) else math.sin(x)
def cos(x):  return x.cos()  if isinstance(x,auto.Diff) else math.cos(x)
def sqrt(x): return x.sqrt() if isinstance(x,auto.Diff) else math.sqrt(x)
def log(x):  return x.ln()   if isinstance(x,auto.Diff) else math.log(x)
def exp(x):  return x.exp()  if isinstance(x,auto.Diff) else math.exp(x)
def fix(x):  return x if isinstance(x,auto.Diff) else diffConst(x)
def pow(x, y):  return x.pow(y)  if isinstance(x,auto.Diff) else math.pow(x, y)
def abs(x): return x.abs() if isinstance(x,auto.Diff) else math.fabs(x)
def const(x): return int(x.primal) if isinstance(x,auto.Diff) else int(x)

pi = diffConst(math.pi)


num_runs = 0
# Dynamically-compiled module handles
active_module = None

# bl_info = {
#     "name": "BiModel",
#     "blender": (2, 80, 0),
#     "category": "Object",
# }

bl_info = {
    "name" : "bimodel",
    "description" : "",
    "blender" : (2, 92, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

# https://rgb.to/html-color-names/1
colors = {
    "blue": (0,0,1),
    "blueviolet": (138/255, 43/255, 226/255),
    "gold": (1, 215/255, 0),
    "burlywood": (222/255, 184/255, 135/255),
    "dodgerblue": (30/255, 144/255, 1),
    "lightseagreen": (32/255, 178/255, 170/255),
}

# https://rgb.to/html-color-names/1
model2colors = {
    "slipper": "blue",
    "chandelier": "gold",
    "chair": "burlywood",
}

tri_mesh_name = "tri_mesh"
trace_mesh_name = "trace_mesh"
sync_mesh_name = "sync_mesh"
def_mesh_name = "def_mesh"
com_mesh_name = "center_of_mass_mesh"

stats_folder = "stats"
obj_folder = "obj"

EPSILON = 1e-4
DECIMALS = 3

SAVE_STATS = False
PRINT_STATS = True
SAVE_SUGGESTIONS = False

LOCALIZATION = False

def to_epsilon(value):
    if value < 0.0:
        raise Exception(f"Value that should be positive is negative {value}")
    elif value < EPSILON:
        return EPSILON
    else:
        return value

def test(lam, number=1):
    import timeit
    return timeit.timeit(lam, number=number)

def comp(l1, l2, args, number=10):
    import timeit
    t1 = timeit.timeit(lambda : l1(args), number=number)
    t2 = timeit.timeit(lambda : l2(args), number=number)

    print(f"T1:{t1}, T2:{t2}")

def flatten(t):
  return [item for sublist in t for item in sublist]

def chunks(lst, n):
  def iterate():
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
  return list(iterate())

def colinear(p1, p2, p3, thresh=0.001):
    vec1 = p2 - p1
    vec2 = p3 - p1
    return colinear_vec(vec1, vec2)

def colinear_vec(v1, v2, thresh=0.001):
    return v1.cross(v2).length < thresh

def colinear_face(f1, f2, thresh=0.001):
    return colinear_vec(f1.normal, f2.normal)

# Todo: move these into their own module
def addVert(p1,p2):
    return (p1[0]+p2[0],p1[1]+p2[1],p1[2]+p2[2])
def subVert(p1,p2):
    return (p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2])
def mulVert(p1,f):
    return (p1[0]*f,p1[1]*f,p1[2]*f)
def divVert(p1,f):
    return (p1[0]/f,p1[1]/f,p1[2]/f)
def addVec(v1,v2):
    return [ a+b for (a,b) in zip(v1,v2) ]
def subVec(v1,v2):
    return [ a-b for (a,b) in zip(v1,v2) ]
def lenVec(v1):
    dx, dy, dz = v1
    return (dx*dx) + (dy*dy) + (dz*dz)
def distSquare(p1, p2):
    return lenVec(subVert(p2, p1))

def normal(p1,p2,p3):
  # print(f"Normal: {p1},{p2},{p3}")
  u = subVec(p2,p1)
  v = subVec(p3,p1)
  nx = u[1]*v[2] - u[2]*v[1]
  ny = u[2]*v[0] - u[0]*v[2]
  nz = u[0]*v[1] - u[1]*v[0]
  length = (nx*nx + ny*ny + nz*nz).sqrt()
  return (nx/length,ny/length,nz/length)


def volume_tri(p1, p2, p3):
    # http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
    # Could be replaced with p1 dot (p2 cross p3)
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123)

def volume(tri_V, tri_F):
    total_vol = 0.0
    for tri in tri_F:
        V = volume_tri(tri_V[tri[0]], tri_V[tri[1]], tri_V[tri[2]])
        total_vol += V
    return total_vol

def center_of_mass(tri_V, tri_F):
    total_vol = 0.0
    total_center = np.zeros(3)
    for tri in tri_F:
        V = volume_tri(tri_V[tri[0]], tri_V[tri[1]], tri_V[tri[2]])
        total_vol += V
        center = (tri_V[tri[0]] + tri_V[tri[1]] + tri_V[tri[2]]) / 4.
        total_center += center * V
    return total_center / total_vol

# TODO: Right now we're autodiffing through area. It's not very fast,
#       especially if there are many triangles in the model.
# https://www.iquilezles.org/blog/?p=1579
# A² = (2ab + 2bc + 2ca – a² – b² – c²)/16
def surface_area_tri(p1, p2, p3):
    a = distSquare(p1, p2)
    b = distSquare(p1, p3)
    c = distSquare(p2, p3)
    A = (2.0*a*b + 2.0*b*c + 2.0*c*a - a*a - b*b - c*c) / 16.0
    return A

def surface_area_float(tri_V, tri_F):
    total_area = 0.0
    for tri in tri_F:
        p1,p2,p3 = tri_V[tri[0]], tri_V[tri[1]], tri_V[tri[2]]
        A = surface_area_tri(p1,p2,p3)
        total_area += sqrt(A)
    return total_area

def squared_surface_area_float(tri_V, tri_F):
    total_area = 0.0
    for tri in tri_F:
        p1,p2,p3 = tri_V[tri[0]], tri_V[tri[1]], tri_V[tri[2]]
        A = surface_area_tri(p1,p2,p3)
        total_area += A
    return total_area

# Todo: fix this to have consistent notation so we can call sqrt(A) even
# when A is Diff.
def surface_area_diff(tri_V, tri_F):
    total_area = 0.0
    for tri in tri_F:
        p1,p2,p3 = tri_V[tri[0]], tri_V[tri[1]], tri_V[tri[2]]
        A = surface_area_tri(p1,p2,p3)
        total_area += A.sqrt()
    return total_area

class KD_cls:
    def __init__(self, bm, verts):
        self.kd = mathutils.kdtree.KDTree(len(verts))
        for vert in verts:
            vec = bm.verts[vert].co
            # print(f"Adding vec: {vert} => {vec}")
            self.kd.insert(vec, vert)
        self.kd.balance()

    def find(self, vector):
        return self.kd.find(vector)

def create_model(name):
    if name not in bpy.data.meshes:
        mesh = bpy.data.meshes.new(name)
    else:
        mesh = bpy.data.meshes[name]

    if name not in bpy.data.objects:
        obj = bpy.data.objects.new(name, mesh)
        bpy.data.collections["Collection"].objects.link( obj )
    else:
        obj = bpy.data.objects[name]

def select_model(name):
    for obj in bpy.data.collections["Collection"].objects:
        if obj.type == 'MESH':
            if not obj.name == name:
                obj.select_set(False)
                if not obj.name.lower().startswith("target"):
                  obj.hide_set(True)
            else:
                obj.select_set(True)
                obj.hide_set(False)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

def save_to_obj(bm, filePath, withNormals=False):
    with open(filePath, 'w') as f:
        f.write(f"# {len(bm.verts)} {len(bm.faces)}\n")

        # Vertex positoins
        for vert in bm.verts:
            vec = vert.co
            f.write(f"v {vec.x:.6f} {vec.z:.6f} {vec.y:.6f}\n")

        # Disable smooth shading
        f.write("s off\n")

        # Normals
        # Vertices are given in winding order so use this for debugging
        if withNormals:
            for face in bm.faces:
                normal = face.normal
                f.write(f"vn {normal.x:.4f} {normal.z:.4f} {normal.y:.4f}\n")

        # Face positions
        for i, face in enumerate(bm.faces):
            f.write(f"f")
            # Blender returns these in counter clockwise order so we need to reverse for obj
            for vert in reversed(face.verts):
                # Indices start at 1
                f.write(f" {vert.index + 1}")
                if withNormals:
                    f.write(f"//{i + 1}")
            f.write("\n")

def clear_edit_mesh():
    me = active_object()
    bmesh.from_edit_mesh(me).clear()
    bmesh.update_edit_mesh(me, True, True)

def update_edit_mesh():
    bmesh.update_edit_mesh(active_object(), True, True)

def edit_mesh():
    data = active_object()
    if not data:
        return None
    bm = bmesh.from_edit_mesh(data)
    ensure_tables(bm)
    return bm

def edit_mesh_to_numpy():
    bm = edit_mesh()
    if not bm:
        return None
    bm = bm.copy()
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
    ensure_tables(bm)
    V = verts_to_numpy(bm)
    F = faces_to_numpy(bm)
    return V, F

def active_object():
    obj = bpy.context.active_object
    if obj.mode == 'OBJECT':
        return None
    return obj.data

def ensure_tables(bm):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

# This will kill any bmesh object obtained from active_mesh()
def save_edit_mesh():
    # NOTE (PAQ): Setting the mode to OBJECT causes the current active edit mesh to be saved.
    update_edit_mesh()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')

def load_mesh_name(name):
    assert name in bpy.data.meshes
    return bpy.data.meshes[name]

def load_mesh(name):
    mesh = load_mesh_name(name)
    return new_bm_from_mesh(mesh)

def new_bm_from_mesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    ensure_tables(bm)
    return bm

def save_mesh(name, bm, hide=None):
    if name not in bpy.data.meshes:
        bpy.data.meshes.new(name)
    mesh = bpy.data.meshes[name]
    bm.to_mesh(mesh)
    if mesh.name not in bpy.data.objects:
        bpy.data.objects.new(mesh.name, mesh)
    obj = bpy.data.objects[mesh.name]
    if obj.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(obj)
    if hide is not None:
        obj.hide_set(hide)
    obj.data.update()
    # print("Saved mesh", name)

def update_active_model(verts):
    me = active_object()
    mesh_verts_from_numpy(me, verts)

def mesh_to_numpy(name):
    me_tri = load_mesh_name(name)
    bm = bmesh.new()
    bm.from_mesh(me_tri)
    # V = mesh_verts_to_numpy(me_tri)
    # F = mesh_faces_to_numpy(me_tri)
    V = np.array([v.co for v in bm.verts], dtype=np.float)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int)
    bm.free()
    return V, F

def mesh_verts_to_numpy(me):
    verts = np.empty(len(me.vertices)*3, dtype=np.float64)
    me.vertices.foreach_get('co', verts)
    return verts.reshape(-1, 3)

def mesh_faces_to_numpy(me, size=3):
    faces = np.empty(len(me.polygons)*3, dtype=np.int32)
    me.polygons.foreach_get('vertices', faces)
    return faces.reshape(-1, size)

def mesh_verts_from_numpy(me, verts):
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    me.vertices.foreach_set('co', verts.flatten())
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

def verts_to_numpy(bm):
    V = np.fromiter((x for v in bm.verts for x in v.co),
                     dtype=np.float64,
                     count=len(bm.verts)*3)
    V.shape = (len(bm.verts), 3)
    return V

# Must be a triangulated mesh
def faces_to_numpy(bm):
    F = np.fromiter((v.index for f in bm.faces for v in f.verts),
                     dtype=np.int32,
                     count=len(bm.faces)*3)
    F.shape = (len(bm.faces), 3)
    return F

def verts_from_numpy(bm, verts):
    def f2(n,v):
        v.co = n
    for v in itertools.starmap(f2, zip(verts, bm.verts)): pass

def bmesh_face_indices(bm):
   return [v.index for face in bm.faces for v in face.verts]


def set_edge_selection(bm, edges, union=False):
    bm.select_mode = {'EDGE'}
    for e in bm.edges:
        e.select = e.index in edges or (union and e.select)
    bm.select_flush_mode()


def set_vert_selection(bm, verts, union=False):
    bm.select_mode = {'VERT'}
    for v in bm.verts:
        v.select = v.index in verts or (union and v.select)
    bm.select_flush_mode()


def set_face_selection(bm, faces, union=False):
    bm.select_mode = {'FACE'}
    for f in bm.faces:
        f.select = f.index in faces or (union and f.select)
    bm.select_flush_mode()


def get_edge_selection(bm):
    return [e.index for e in bm.edges if e.select]

def get_vert_selection(bm):
    return [v.index for v in bm.verts if v.select]

def get_face_selection(bm):
    return [f.index for f in bm.faces if f.select]

def neighbors(bm, v):
    return [edge.other_vert(bm.verts[v]).index for edge in bm.verts[v].link_edges]

# Use BFS to cluster faces that share edges
def cluster_adjacent_bm(bm, faces):
    face_clusters = []
    while len(faces) > 0:
        q = faces[0:1]
        face_clusters.append(q)
        faces = faces[1:]
        for f in q:
            ff = bm.faces[f]
            for e in ff.edges:
                for adj in e.link_faces:
                    if adj.index == f:
                        continue
                    # Ignore adjacent faces that are not colinear
                    if (adj.index in faces) and colinear_face(ff, adj):
                        q.append(adj.index)
                        faces.remove(adj.index)
    return face_clusters

def to_sympy_array(vector):
    return Array([vector[0], vector[1], vector[2]])

def is_number(node):
    if isinstance(node, ast.Num):
        return True
    elif isinstance(node, ast.UnaryOp):
        return is_number(node.operand)
    else:
        return False

class Location:
    __slots__ = ['lineno', 'col']

    def __repr__(self):
        return f"Ln {self.lineno}, Col {self.col}"

    def __init__(self, node):
        assert isinstance(node, ast.Num) or isinstance(node, ast.UnaryOp), "Locations must be built from number nodes."

        self.lineno = node.lineno - 1
        self.col = node.col_offset

def get_text_area():
    for area in bpy.context.window.screen.areas:
        if area.type == 'TEXT_EDITOR':
            return area
    raise Exception("No screen area found")

class TextManager:
    __slots__ = ['text', 'parameters', 'context', 'line_locations']

    def __init__(self, area):
        self.text = area.spaces.active.text
        self.parameters = {}
        self.line_locations = {}
        self.context = {'area': area}

    def name(self):
        return self.text.name

    def safe_name(self):
        return self.text.name.replace(".", "_")

    def program(self):
        return self.text.as_string()

    def add_parameter(self, name, location):
        self.parameters[name] = location
        if location.lineno not in self.line_locations:
            self.line_locations[location.lineno] = []
        self.line_locations[location.lineno].append(location)

    def update_program(self, params):
        for name, value in zip(self.parameters.keys(), params):
            self.set_parameter(name, value)


    def set_parameter(self, name: str, value: float):
        text = self.context['area'].spaces.active.text
        loc = self.parameters[name]
        line = text.lines[loc.lineno].body[loc.col:]
        number_text = re.match(r"[+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*)(?:[eE][+-]?\d+)?", line).group(0)
        number_len = len(number_text)
        text.select_set(line_start=loc.lineno, char_start=loc.col, line_end=loc.lineno, char_end=loc.col+number_len)
        # Make sure floats don't have too many decimal places. No need to format to constant length
        new_text = "{}".format(round(value, DECIMALS))
        new_text_len = len(new_text)
        diff_len = new_text_len - number_len
        text.write(new_text)
        for line_loc in self.line_locations[loc.lineno]:
            if line_loc.col > loc.col:
                line_loc.col += diff_len

class BIMODEL_OT_Run(bpy.types.Operator):
    bl_idname = "bimodel.run"
    bl_label = "Run"
    bl_description = "Runs shallow embedding"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        tm = TextManager(get_text_area())
        program = tm.program()
        ldict = {}
        dsl.run(program, tm.name(), ldict)
        print(ldict)
        return {"FINISHED"}

def find_matches(traces, thresh=0.01):
    save_edit_mesh()
    bm = edit_mesh()
    verts = get_vert_selection(bm)
    kd = KD_cls(bm, verts)

    matches = []
    for trace in traces:
        vec = valueOf(trace)
        co, vert, dist = kd.find(vec)
        assert dist < thresh, f"Match error: VERT({vert}) => {co}, TRACE({trace}) => {vec}, DIST({dist}) > THRESH({thresh})"
        matches.append((trace, vert))

    return matches

def sequential_matches(traces):
    save_edit_mesh()
    bm = edit_mesh()

    total = len(bm.verts)
    new = len(traces)
    return [(trace, index + total - new) for index, trace in enumerate(traces)]

class ModelRef:
    model = None

    def greater(self, e1, e2):
        if valueOf(e1) < valueOf(e2):
            raise Exception(f"Innequality incorrect state! [e1={valueOf(e1)} < e2 = {valueOf(e2)}]")
        if isinstance(e1, auto.Diff) or isinstance(e2, auto.Diff):
            self.model.inequality_constraints.append(e1 - e2)

    def positive(self, expr):
        if (isinstance(expr, tuple)):
            for e in expr:
                self.greater(e, EPSILON)
        else:
            self.greater(expr, EPSILON)

    def toDiff(self, var):
        if isinstance(var, auto.Diff):
            return var
        elif isinstance(var, numbers.Number):
            return diffConst(var)
        elif isinstance(var, tuple):
            return tuple(map(self.toDiff, var))
        else:
            raise Exception(f"Unexpected type {var}")

    def centroid(self, verts):
        num = len(verts)
        sum = self.model.traces[verts[0]]
        for vert in verts[1:]:
            sum = addVert(sum, self.model.traces[vert])
        center = mulVert(sum, 1.0/float(num))
        return center

    def swap(self, bm, v1, v2):
        bm.verts[v1].index = v2
        bm.verts[v2].index = v1
        bm.verts.sort()
        bm.verts.ensure_lookup_table()

        model = self.model
        if v1 in model.vert_objects:
            v1_name, v1_ind = model.vert_objects[v1]
            model.objects[v1_name].verts[v1_ind] = v2
        if v2 in model.vert_objects:
            v2_name, v2_ind = model.vert_objects[v2]
            model.objects[v2_name].verts[v2_ind] = v1

        if v2 in model.vert_objects:
            model.vert_objects[v1] = (v2_name, v2_ind)

        if v1 in model.vert_objects:
            model.vert_objects[v2] = (v1_name, v1_ind)

        model.traces[v1], model.traces[v2] = model.traces[v2], model.traces[v1]

    def move_to_front(self, bm, verts):
        change = len(bm.verts)
        verts.sort()
        verts.reverse()
        for i, vert in enumerate(verts):
            self.swap(bm, vert, change - i - 1)

    def delete_last(self, bm, count):
        change = len(bm.verts)
        for i in range(count):
            self.delete(change - i - 1)

    def delete(self, vert):
        model = self.model
        if vert in model.vert_objects:
            name, ind = model.vert_objects[vert]
            del model.objects[name].verts[ind]
            del model.vert_objects[vert]
        del model.traces[vert]

class Clamp(ModelRef):
    def __init__(self, value=None, lower=None, upper=None):
        value = self.toDiff(value)
        lower = self.toDiff(lower)
        upper = self.toDiff(upper)

        if value is None:
            raise Exception("Clamp() needs value to bound, found `None`.")
        if lower is None and upper is None:
            raise Exception("Clamp() needs either a `lower()` or `upper()` bound to work.")
        if lower is not None:
            self.greater(value, lower)
        if upper is not None:
            self.greater(upper, value)

class Object(ModelRef):
    def __init__(self, name, **kwargs):
        super().__init__()
        self.name = name
        self.verts = {}
        
    def __str__(self):
        return f"{self.name}[{self.verts}]"

    def __repr__(self):
        return self.__str__()

    def _add_vert(self, index, vert):
        if index in self.verts:
            raise Exception(f"INDEX[{index}] is already assigned for object {self.name}.")
        self.verts[index] = vert

    def add_matches(self, trace_matches, name, update=False, check=False):
        save_edit_mesh()
        bm = edit_mesh()
        model = self.model
        if name and name not in model.objects:
                model.objects[name] = self
        for index, (trace, vert) in enumerate(trace_matches):
            if check:
                co = bm.verts[vert].co
                vec = valueOf(trace)
                dist = (co - vec).length
                thresh = 0.01
                assert dist < thresh, f"Match error: VERT({vert}) => {co}, TRACE({trace}) => {vec}, DIST({dist}) > THRESH({thresh})"

            model.traces[vert] = trace
            if name:
                self._add_vert(index, vert)
                if vert in model.vert_objects:
                    raise Exception(f"VERT[{vert}] already assigned.")
                model.vert_objects[vert] = (name, index)

    def v(self, indices):
        return [self.verts[i] for i in indices]

    def x(self, index):
        return self.model.traces[self.verts[index]][0]
    def y(self, index):
        return self.model.traces[self.verts[index]][1]
    def z(self, index):
        return self.model.traces[self.verts[index]][2]
    def ref(self, index):
        pt = self.model.traces[self.verts[index]]
        ref = RefPoint(pt[0],pt[1],pt[2])
        return ref

    def all(self):
        return list(self.verts.values())

class Cube(Object):
    def __init__(self, name, size=1., location=(0, 0, 0), **kwargs):
        super().__init__(name, **kwargs)
        self.positive(size)
        size = self.toDiff(size)
        location = self.toDiff(location)

        bpy.ops.mesh.primitive_cube_add(size=valueOf(size), location=valueOf(location))
        # It's important that we compute these like so, since it results in significantly
        # greater sharing of subexpressions in the autodiff tree than we would otherwise have.
        e = size * 0.5
        ne = 0-e
        ex = e +   location[0]
        ey = e +   location[1]
        ez = e +   location[2]
        nex = ne + location[0]
        ney = ne + location[1]
        nez = ne + location[2]

        traces = [
          [nex, ney, nez],
          [nex, ney,  ez],
          [nex,  ey, nez],
          [nex,  ey,  ez],
          [ ex, ney, nez],
          [ ex, ney,  ez],
          [ ex,  ey, nez],
          [ ex,  ey,  ez]]

        matches = sequential_matches(traces)
        self.add_matches(matches, name)

class Box(Object):
    def __init__(self, name, dims=(1.0, 1.0, 1.0), base=(0.0, 0.0, 0.0), **kwargs):
        super().__init__(name, **kwargs)
        self.positive(dims)
        dims = self.toDiff(dims)
        base = self.toDiff(base)

        cen = (base[0]+dims[0]*0.5,base[1]+dims[1]*0.5,base[2]+dims[2]*0.5)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=valueOf(cen))
        bpy.ops.transform.resize(value=valueOf(dims))

        nx,ny,nz = base[0],base[1],base[2]
        px,py,pz = nx+dims[0],ny+dims[1],nz+dims[2]

        traces = [
          [nx,ny,nz],
          [nx,ny,pz],
          [nx,py,nz],
          [nx,py,pz],
          [px,ny,nz],
          [px,ny,pz],
          [px,py,nz],
          [px,py,pz]]
        matches = sequential_matches(traces)
        self.add_matches(matches, name)

class Grid(Object):

    def __init__(self, name, size=1.0, location=(0.0, 0.0, 0.0), subs=(2, 2), **kwargs):
        super().__init__(name, **kwargs)
        self.positive(size)
        size = self.toDiff(size)
        location = self.toDiff(location)
        x_subs, y_subs = subs

        bpy.ops.mesh.primitive_grid_add(x_subdivisions=x_subs, y_subdivisions=y_subs, size=valueOf(size), location=valueOf(location))

        traces = []
        offset = subVert(location, (size * 0.5, size * 0.5, 0.0))
        x_div = size / (x_subs - 1)
        y_div = size / (y_subs - 1)
        for y in range(y_subs):
            for x in range(x_subs):
                traces.append(addVert(offset,(x * x_div, y * y_div, 0.0)))

        matches = sequential_matches(traces)
        self.add_matches(matches, name)

class Cylinder(Object):
    
    def __init__(self, name, radius=1.0, depth=2.0, location=(0.0, 0.0, 0.0), subs=8, fill="NGON", **kwargs):
        super().__init__(name, **kwargs)
        self.positive(radius)
        self.positive(depth)
        radius = self.toDiff(radius)
        depth = self.toDiff(depth)
        location = self.toDiff(location)

        traces = []
        offset = location
        top = depth * 0.5
        bot = -top
        if fill == "TRIFAN":
            traces.append([offset[0], bot+offset[1], offset[2]])
            traces.append([offset[0], top+offset[1], offset[2]])

        theta = 0
        tau = 2.0 * math.pi
        pi2 = math.pi / 2.0
        for i in range(subs):
            theta = tau * i / subs
            x = -radius * cos(pi2+theta)
            y =  radius * sin(pi2+theta)
            traces.append(addVec([x, y, bot], offset))
            traces.append(addVec([x, y, top], offset))

        bpy.ops.mesh.primitive_cylinder_add(
          location=valueOf(location),
          vertices=subs,
          radius=valueOf(radius),
          depth=valueOf(depth),
          end_fill_type=fill)

        matches = sequential_matches(traces)
        self.add_matches(matches, name)

class Sphere(Object):

    def __init__(self, name, radius=1.0, location=(0.0, 0.0, 0.0), segments=20, ring_count=16, **kwargs):
        super().__init__(name, **kwargs)
        self.positive(radius)
        radius = self.toDiff(radius)
        location = self.toDiff(location)

        traces = []
        offset = location

        # Stage a bunch of nodes so we can reuse subexprs.
        tau = math.pi*2
        pi2 = math.pi/2
        thetas = [pi2+(tau*i)/segments for i in range(segments)]
        phis = [(math.pi*i)/(ring_count) for i in range(1, ring_count)]
        theta_map = [ (sin(theta),cos(theta)) for theta in thetas ]
        phi_map = [ (sin(phi),cos(phi)) for phi in phis ]

        for (sint,cost) in theta_map:
            for (sinp,cosp) in phi_map:
                traces.append(addVec([radius*sinp*cost, radius*sinp*sint, radius*cosp], offset))

        traces.append(addVec([0, 0, radius], offset))
        traces.append(addVec([0, 0, -radius], offset))

        bpy.ops.mesh.primitive_uv_sphere_add(
            location=valueOf(location),
            radius=valueOf(radius),
            segments=segments,
            ring_count=ring_count)

        matches = find_matches(traces)
        self.add_matches(matches, name)

class Circle(Object):

    def __init__(self, name, radius=1.0, location=(0.0, 0.0, 0.0), fill="NGON", subs=8, **kwargs):
        super().__init__(name, **kwargs)
        self.positive(radius)
        radius = self.toDiff(radius)
        location = self.toDiff(location)

        traces = []
        offset = location
        if fill == "TRIFAN":
            traces.append([offset[0], offset[1], offset[2]])

        tau = 2.0 * math.pi
        pi2 = math.pi / 2.0
        theta = pi2
        for i in range(subs):
            theta = pi2-tau * i / subs
            x = radius * cos(theta)
            y =  radius * sin(theta)
            traces.append(addVec([x, y, 0], offset))

        bpy.ops.mesh.primitive_circle_add(
          location=valueOf(location),
          vertices=subs,
          radius=valueOf(radius),
          fill_type=fill)

        matches = sequential_matches(traces)
        self.add_matches(matches, name)

class Modifier:
    def select(self, select=None, union=False, all=None, **kwargs):
        bm = edit_mesh()
        if all:
            bpy.ops.mesh.select_all(action=all)
        elif select:
            set_vert_selection(bm, select, union)

class Select(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)

class Loop(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.loop_multi_select(ring=False)

class Ring(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.loop_multi_select(ring=True)


class Region(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.loop_to_region()

class Boundry(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.region_to_loop()

class SPath(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.shortest_path_select(edge_mode='SELECT')

class Connected(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.select_linked(delimit=set())

        
class Delete(ModelRef, Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)

        bm = edit_mesh()
        verts = get_vert_selection(bm)
        self.move_to_front(bm, verts)
        self.delete_last(bm, len(verts))
        bpy.ops.mesh.delete(type='VERT')

class Resize(ModelRef, Modifier):

    def __init__(self, value=(1.0, 1.0, 1.0), center=None, **kwargs):
        self.select(**kwargs)
        self.positive(value)
        value = self.toDiff(value)

        bm = edit_mesh()
        verts = get_vert_selection(bm)
        sx,sy,sz = value
        if center:
            cx, cy, cz = center
            vert_objs = [bm.verts[vert] for vert in verts]
            bmesh.ops.translate(bm, verts=vert_objs, vec=(-valueOf(cx), -valueOf(cy), -valueOf(cz)))
            bmesh.ops.scale(bm, verts=vert_objs, vec=(valueOf(sx), valueOf(sy), valueOf(sz)) )
            bmesh.ops.translate(bm, verts=vert_objs, vec=(valueOf(cx), valueOf(cy), valueOf(cz)))
        else:
            bpy.ops.transform.resize(value=valueOf((sx, sy, sz)))

        if center == None:
            center = self.centroid(verts)

        for vert in verts:
            trace = self.model.traces[vert]
            dx, dy, dz = subVert(trace, center)
            self.model.traces[vert] = addVert(center, (dx * sx, dy * sy, dz * sz))

class Translate(ModelRef, Modifier):
    def __init__(self, value=(0.0, 0.0, 0.0), **kwargs):
        self.select(**kwargs)
        value = self.toDiff(value)

        bpy.ops.transform.translate(value=valueOf(value))
        bm = edit_mesh()
        offset = value
        verts = get_vert_selection(bm)
        for vert in verts:
            self.model.traces[vert] = addVert(self.model.traces[vert],offset)    

class Rotate(ModelRef, Modifier):
    def __init__(self, theta=0.0, axis=None, center=None, **kwargs):
        self.select(**kwargs)
        theta = self.toDiff(theta)

        bm = edit_mesh()
        verts = get_vert_selection(bm)

        if center == None:
            center = self.centroid(verts)

        cosθ = theta.cos()
        sinθ = theta.sin()
        if axis != None:
            mat_rot = mathutils.Matrix.Rotation(valueOf(theta), 4, axis)
            bmesh.ops.rotate(bm, verts=[bm.verts[vert] for vert in verts], cent=valueOf(center), matrix=mat_rot)
            update_edit_mesh()
            cx, cy, cz = center
            if axis == 'X':
                for vert in verts:
                    x, y, z = self.model.traces[vert]
                    ycy, zcz = y-cy, z-cz
                    Ry = ycy * cosθ - zcz * sinθ + cy
                    Rz = zcz * cosθ + ycy * sinθ + cz
                    self.model.traces[vert] = [x, Ry, Rz]
            elif axis == 'Y':
                for vert in verts:
                    x, y, z = self.model.traces[vert]
                    xcx, zcz = x-cx, z-cz
                    Rx = xcx * cosθ + zcz * sinθ + cx
                    Rz = zcz * cosθ - xcx * sinθ + cz
                    self.model.traces[vert] = [Rx, y, Rz]
            elif axis == 'Z':
                for vert in verts:
                    x, y, z = self.model.traces[vert]
                    xcx, ycy = x-cx, y-cy
                    Rx = xcx * cosθ - ycy * sinθ + cx
                    Ry = ycy * cosθ + xcx * sinθ + cy
                    self.model.traces[vert] = [Rx, Ry, z]
        else:
            raise Exception(f"Axis not set, valid are 'X', 'Y', 'Z'.")

class Bridge(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.bridge_edge_loops(type='SINGLE', use_merge=False, merge_factor=0.5)

class DeleteEdge(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.delete(type='EDGE')

class DeleteFace(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.delete(type='FACE')

class Fill(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.edge_face_add()

class Flip(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.flip_normals()

class Recalculate(Modifier):
    def __init__(self, inside=False, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.normals_make_consistent(inside=inside)

class Triangulate(Modifier):
    def __init__(self, **kwargs):
        self.select(**kwargs)
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

class ModifierObject(Object, Modifier):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.select(**kwargs)

class Extrude(ModifierObject):
    def __init__(self, name, length=1, dynamic=False, **kwargs):
        super().__init__(name, **kwargs)
        self.positive(length)
        length = self.toDiff(length)

        bm = edit_mesh()
        face_indices = get_face_selection(bm)
        if valueOf(length) < 0:
            raise Exception(f"Extrude is negative: {valueOf(length)} < 0")
        if len(face_indices) < 1:
            raise Exception(f"Extrude is a noop")
        # Group face indices by connections
        # Also checks that faces are colinear
        face_clusters = cluster_adjacent_bm(bm, face_indices)

        # Calculate normals.
        for _, inds in enumerate(face_clusters):
            bm = edit_mesh()
            set_face_selection(bm, inds)
            cluster_face = bm.faces[inds[0]]

            current_selection = get_vert_selection(bm)
            # Account for normal changing.
            if dynamic:
                cluster_verts = [self.model.traces[v.index] for v in cluster_face.verts][:3]
                normal_vec = normal(*cluster_verts)
            # Otherwise, use the original normal for everything.
            # (This is OK if we know it's not going to change.)
            else:
                cluster_normal = cluster_face.normal
                normal_vec = map(diffConst,cluster_normal)

            traces = []
            normal_offset = [k*length for k in normal_vec]
            for vert in current_selection:
                traces.append(addVec(self.model.traces[vert],normal_offset))

            # Make sure the current indices dont change after the extrude
            save_edit_mesh()
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
            bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={
                'value':(0.0, 0.0, valueOf(length)),
                'orient_type':'NORMAL',
                'orient_matrix_type':'NORMAL',
                'constraint_axis':(False, False, True)
            })

        matches = find_matches(traces)
        self.add_matches(matches, name)

class Chamfer(ModifierObject):

    def __init__(self, name, length=0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.positive(length)
        length = self.toDiff(length)

        bm = edit_mesh()
        verts = get_vert_selection(bm)
        self.move_to_front(bm, verts)
        verts = get_vert_selection(bm)
        edges = get_edge_selection(bm)
        distances = []
        traces = []

        for vert in verts:
            for edge in bm.verts[vert].link_edges:
                if edge.index not in edges:
                    if edge.calc_length() < valueOf(length):
                        raise Exception(f"Chamfer length greater than minimum {edge.calc_length()}")
                    v1 = vert
                    v2 = edge.other_vert(bm.verts[v1]).index

                    vec = subVec(self.model.traces[v2], self.model.traces[v1])
                    dist = sqrt(lenVec(vec))
                    new = addVec(self.model.traces[v1], mulVert(vec, length/dist))
                    traces.append((new, v2))
                    distances.append(dist)

        self.delete_last(bm, len(verts))
        bpy.ops.mesh.bevel(offset=valueOf(length), offset_type="ABSOLUTE")

        bm = edit_mesh()
        new_verts = get_vert_selection(bm)

        trace_matches = []
        for trace, border in traces:
            connected = neighbors(bm, border)
            new_vert = [v for v in connected if v in new_verts][0]
            trace_matches.append((trace, new_vert))

        self.add_matches(trace_matches, name, update=True)
        for edge_length in distances:
            self.greater(edge_length, length)

class RefPoint:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __str__(self):
    return f"(X: {self.x.__repr__()}, Y: {self.y.__repr__()}, Z: {self.z.__repr__()})"

  def __repr__(self):
    return self.__str__()

class ArrayNames(ast.NodeTransformer):
    __slots__ = ['loop_vars']

    def __init__(self, loop_vars):
        self.loop_vars = loop_vars

    def visit_Name(self, node):
        assert '_' not in node.id , f"Names cannot contain underscores, {node.id}."
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_Subscript(self, node):
        assert isinstance(node.value, ast.Name), "Only first level subscripts supported, use tuple for multi dim."
        name = node.value.id
        index = eval(astunparse.unparse(node.slice), globals().update(self.loop_vars))
        if isinstance(index, tuple):
            index = "_".join(map(str, index))
        return ast.Name(id=f"{name}_{index}")

    @staticmethod
    def remake_sub(name):
        if "_" in name:
            return name.replace("_", "[", 1).replace("_", ", ")+"]"
        else:
            return name

class Parameter:

    def __init__(self, name, value, lineno, col_offset):
        self.name = name
        self.value = value
        self.lineno = lineno
        self.col_offset = col_offset


class BiModel(ast.NodeVisitor):
    last_prog = None

    @classmethod
    def has_model(cls):
        return cls.last_prog is not None

    @classmethod
    def reset(cls):
      cls.last_prog = None

    def __init__(self, text_manager):
        self.text_manager = text_manager
        self.assignments = {}
        self.symbols = []
        self.ctx = auto.DiffContext()

        self.traces = {}
        self.inequality_constraints = []

        self.objects = {}
        self.vert_objects = {}

        self.loop_vars = {}

        self.current_params = None

    def in_loop(self):
        return bool(self.loop_vars)

    def parseProgram(self):
        root = ast.parse(self.text_manager.program())
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        ModelRef.model = self
        if isinstance(root.body[0], ast.Import):
            print("Running...")
            ldict = {}
            exec(compile(root, self.text_manager.name(), 'exec'), globals(), ldict)
        else:
            print("Parsing...")
            self.visit(root)

    def clean_subscript(self, node):
        import copy
        # TODO (PAQ): We require a deep copy here since we may evaluate the subscripts at multiple times per loop.
        cleaner = ArrayNames(self.loop_vars)
        new_node = cleaner.visit(copy.deepcopy(node))
        # ast.fix_missin@g_locations(new_node)
        return new_node

    def parseParameter(self, node, name, positive=False):
        isParameter = False
        if is_number(node):
            value = diffConst(ast.literal_eval(node)) # Lift the parameter as a constant
            if name:
                value = diffVar(name, value)           # Make it into a differentiable parameter
                self.symbols.append(value)
                isParameter = True
            else:
                positive = False # Don't need to clamp constants to be positive
        else:
            expr = astunparse.unparse(self.clean_subscript(node))
            result = eval(expr, globals().update(dict(self.assignments, **self.loop_vars, **self.objects)))
            value = lift(result)

        # if positive:
        #     self.positive(value)

        return (value, isParameter)

    def parseOperation(self, node, name=None):
        func_name = node.func.id
        keywords = self.parseKeywords(node)
        if func_name in {'Cube'}:
            self.parseCube(keywords, name)
        elif func_name in {'Box'}:
            self.parseBox(keywords, name)
        elif func_name in {'Chamfer'}:
            self.parseChamfer(keywords, name)
        elif func_name in {'Extrude', 'ExtrudeFace'}:
            self.parseExtrude(keywords, name)
        elif func_name in {'Grid'}:
            self.parseGrid(keywords, name)
        elif func_name in {'Cylinder'}:
            self.parseCylinder(keywords, name)
        elif func_name in {'Sphere'}:
            self.parseSphere(keywords, name)
        elif func_name in {'Circle'}:
            self.parseCircle(keywords, name)
        elif func_name in {'Resize'}:
            self.parseResize(keywords)
        elif func_name in {'Rotate'}:
            self.parseRotate(keywords)
        elif func_name in {'Translate'}:
            self.parseTranslate(keywords)
        elif func_name in {'Select'}:
            self.parseSelect(keywords)
        elif func_name in {'Delete', 'DeleteVert'}:
            self.parseDelete(keywords)
        elif func_name in {'Bridge'}:
            self.parseBridge(keywords)
        elif func_name in {'Clamp'}:
            self.parseClamp(keywords)
        elif func_name in {'DeleteEdge'}:
            self.parseDeleteEdge(keywords)
        elif func_name in {'DeleteFace'}:
            self.parseDeleteFace(keywords)
        elif func_name in {'Fill'}:
            self.parseFill(keywords)
        elif func_name in {'Triangulate'}:
            self.parseTriangulate(keywords)
        elif func_name in {'Flip'}:
            self.parseFlip(keywords)
        elif func_name in {'Path'}:
            self.parsePath(keywords)
        elif func_name in {'Connected'}:
            self.parseConnected(keywords)
        elif func_name in {'Loop'}:
            self.parseLoop(keywords)
        elif func_name in {'Ring'}:
            self.parseRing(keywords)
        elif func_name in {'Region'}:
            self.parseRegion(keywords)
        elif func_name in {'Boundry'}:
            self.parseBoundry(keywords)

        else:
            raise Exception(f"Unhandled operation encountered: {func_name}")

    def parseAssign(self, node, name):
        if name in self.assignments:
            raise Exception(f"Assigning to previously bound parameter {name} = {astunparse.unparse(node)}")

        value, isParameter = self.parseParameter(node, name if name[0] is not '_' else None)
        self.assignments[name] = value
        if isParameter:
            self.text_manager.add_parameter(name, Location(node))

    def visit_Assign(self, node):
        target = node.targets[0]
        value = node.value

        if not isinstance(target, ast.Tuple):
            pairs = [(target, value)]
        else:
            pairs = zip(target.elts, value.elts)

        for target, value in pairs:
            name = self.clean_subscript(target).id

            if name[0].isupper():
                self.parseOperation(value, name)
            else:
                self.parseAssign(value, name)

    def visit_Call(self, node):
        self.parseOperation(node)

    def visit_For(self, node):
        # print(f"Looping var {node.target.id}")
        steps = eval(astunparse.unparse(node.iter), globals().update(self.assignments))
        for loop_step in steps:
            # print(f"{node.target.id} => {loop_step}")
            if loop_step % 1000 == 0:
                print("Steps", loop_step)
            self.loop_vars[node.target.id] = loop_step
            for line in node.body:
                self.visit(line)
            # ast.NodeVisitor.generic_visit(self, node.body)
        del self.loop_vars[node.target.id]


    def parseLiteral(self, keywords, name, default):
        if name in keywords:
            keywords[name] = ast.literal_eval(keywords[name])
        # else:
        #     keywords[name] = default

    def parseSelectArgs(self, keywords):
        select = keywords.get("select", None)
        if select:
            expr = self.clean_subscript(select)
            verts = eval(astunparse.unparse(expr), globals().update({**self.loop_vars, **self.objects}))
            assert isinstance(verts, list), f"Selection is not a list. {select.lineno}"
            keywords["select"] = verts
        self.parseLiteral(keywords, "union", False)
        self.parseLiteral(keywords, "all", None)

    def parseArg(self, keywords, name, default, positive=False):
        if name in keywords:
            keywords[name] = self.parseParameter(keywords[name], None, positive)[0]
        # elif default == None:
        #     keywords[name] = None
        # elif isinstance(default, numbers.Number):
        #     keywords[name] = diffConst(default)
        # elif isinstance(default,tuple):
        #     keywords[name] = tuple(map(diffConst,default))
        # else:
        #     raise Exception(f"Expected argument to be numeric or tuple type, instead got {type(default)}")

    def parseKeywords(self, node):
        keywords = {}
        assert isinstance(node, ast.Call)
        for kw in node.keywords:
            keywords[kw.arg] = kw.value
        return keywords

    def parseSelect(self, keywords):
        self.parseSelectArgs(keywords)
        Select(**keywords)

    def parsePath(self, keywords):
        self.parseSelectArgs(keywords)
        SPath(**keywords)

    def parseConnected(self, keywords):
        self.parseSelectArgs(keywords)
        Loop(**keywords)

    def parseLoop(self, keywords):
        self.parseSelectArgs(keywords)
        Loop(**keywords)

    def parseRing(self, keywords):
        self.parseSelectArgs(keywords)
        Ring(**keywords)

    def parseRegion(self, keywords):
        self.parseSelectArgs(keywords)
        Region(**keywords)

    def parseBoundry(self, keywords):
        self.parseSelectArgs(keywords)
        Boundry(**keywords)

    def parseBox(self, keywords, name):
        self.parseArg(keywords, "dims", (1.0,1.0,1.0))
        self.parseArg(keywords, "base", (0.0, 0.0, 0.0))
        Box(name, **keywords)

    def parseCube(self, keywords, name):
        self.parseArg(keywords, "size", 1.0, positive=True)
        self.parseArg(keywords, "location", (0.0, 0.0, 0.0), positive=False)
        Cube(name, **keywords)

    def parseGrid(self, keywords, name):
        self.parseArg(keywords, "size", 1.0, positive=True)
        self.parseArg(keywords, "location", (0.0, 0.0, 0.0), positive=False)
        self.parseLiteral(keywords, "subs", (2, 2))
        Grid(name, **keywords)

    def parseCylinder(self, keywords, name):
        self.parseArg(keywords, "radius", 1.0, positive=True)
        self.parseArg(keywords, "depth",  2.0, positive=True)
        self.parseArg(keywords, "location", (0.0, 0.0, 0.0), positive=False)
        self.parseLiteral(keywords, "subs", 8)
        self.parseLiteral(keywords, "fill", "NGON") # fan or ngon
        Cylinder(name, **keywords)

    def parseSphere(self, keywords, name):
        self.parseArg(keywords, "radius", 1.0, positive=True)
        self.parseArg(keywords, "location", (0.0, 0.0, 0.0), positive=False)
        self.parseLiteral(keywords, "segments", 20)
        self.parseLiteral(keywords, "ring_count", 16) # fan or ngon
        Sphere(name, **keywords)

    def parseCircle(self, keywords, name):
        self.parseArg(keywords, "radius", 1.0, positive=True)
        self.parseArg(keywords, "location", (0.0, 0.0, 0.0), positive=False)
        self.parseLiteral(keywords, "fill", "NGON") # fan or ngon
        self.parseLiteral(keywords, "subs", 8)
        Circle(name, **keywords)   

    def parseExtrude(self, keywords, name):
        self.parseSelectArgs(keywords)
        self.parseArg(keywords, "length", 1.0, positive=True)
        self.parseLiteral(keywords, "dynamic", False)
        Extrude(name, **keywords)

    def parseDelete(self, keywords):
        self.parseSelectArgs(keywords)
        Delete(**keywords)

    def parseChamfer(self, keywords, name):
        self.parseSelectArgs(keywords)
        self.parseArg(keywords, "length", 0.5, positive=True)
        Chamfer(name, **keywords)

    def parseResize(self, keywords):
        self.parseSelectArgs(keywords)
        self.parseArg(keywords, "value", (1.0, 1.0, 1.0), positive=True)
        self.parseArg(keywords, "center", None)
        Resize(**keywords)
        
    def parseTranslate(self, keywords):
        self.parseSelectArgs(keywords)
        self.parseArg(keywords, "value", (0.0, 0.0, 0.0))
        Translate(**keywords)

    def parseRotate(self, keywords):
        self.parseSelectArgs(keywords)
        self.parseArg(keywords, "theta", 0.0)
        self.parseLiteral(keywords, "axis", None) # Options: 'X', 'Y', 'Z'
        self.parseArg(keywords, "center", None)
        Rotate(**keywords)

    def parseBridge(self, keywords):
        self.parseSelectArgs(keywords)
        Bridge(**keywords)

    def parseClamp(self, keywords):
        self.parseArg(keywords, "value", None)  
        self.parseArg(keywords, "lower", None)
        self.parseArg(keywords, "upper", None)
        Clamp(**keywords)

    def parseDeleteEdge(self, keywords):
        self.parseSelectArgs(keywords)
        DeleteEdge(**keywords)

    def parseDeleteFace(self, keywords):
        self.parseSelectArgs(keywords)
        DeleteFace(**keywords)

    def parseFill(self, keywords):
        self.parseSelectArgs(keywords)
        Fill(**keywords)

    def parseTriangulate(self, keywords):
        self.parseSelectArgs(keywords)
        Triangulate(**keywords)

    def parseFlip(self, keywords):
        self.parseSelectArgs(keywords)
        Flip(**keywords)

    def parseRecalculate(self, keywords):
        self.parseSelectArgs(keywords)
        self.parseLiteral(keywords, "inside", False)
        Recalculate(**keywords)

    def trace_values(self, params):
        roots = self.symbols
        # This necessarily updates the parameters in our computation graph too :)
        self.ctx.update({roots[i]: p for (i,p) in enumerate(params) })
        return [valueOf(self.traces[i]) for i in range(len(self.traces))]

    def update_params(self, params):
        self.current_params = params.copy()
        # self.trace_values(params) # NEED TO DO THIS TO UPDATE THE MODEL SYMBOLS!
        new_vecs =  Opt.trace_vertices(self.current_params)
        update_active_model(new_vecs)
        # bm = edit_mesh()
        # for i, vert in enumerate(bm.verts):
        #     vert.co = Vector(new_vecs[i])
        self.text_manager.update_program(self.current_params)


def profile(func):
    @functools.wraps(func)
    def wrapped(self, context):
        profiler = cProfile.Profile()
        profiler.enable()
        value = func(self, context)
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('tottime')

        if PRINT_STATS:
            stats.print_stats(10)

        if SAVE_STATS:
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
            Path(f"./{stats_folder}").mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(f"{stats_folder}/{func.__qualname__}")

        return value
    return wrapped

class BIMODEL_OT_Interpret(Operator):
    """Parse model script"""
    bl_idname = "bimodel.interpret"
    bl_label = "Interpret script"
    bl_options = {'REGISTER', 'UNDO'}


    @profile
    def execute(self, context):

        if not bpy.data.filepath:
            print("WARNING FILE NOT SAVED!")
        else:
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

        BiModel.reset()
        Opt.reset()

        tm = TextManager(get_text_area())

        create_model(tm.name())
        select_model(tm.name())
        clear_edit_mesh()

        # auto.Diff.uuid = 0
        model = BiModel(tm)
        try:
            model.parseProgram()
        except Exception as e:
            # NOTE (PAQ): Saving the mesh after an exception fixes the segfault.
            # The broblem seemed to stem from:
            # 1) The mesh is updated but an exception raises us out of execute.
            # 2) The error is fixed and the mesh is saved.
            # 3) If another exception happens blender segfauls when setting the state.
            save_edit_mesh()

            raise e
        else:
            BiModel.last_prog = model

            # Save the mesh back to obj.data
            bpy.ops.object.mode_set(mode='OBJECT')

            traced = bpy.data.objects[tm.name()]
            bm = bmesh.new()
            bm.from_mesh(traced.data)
            ensure_tables(bm)
            save_mesh(trace_mesh_name, bm)
            save_mesh(def_mesh_name, bm)
            bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
            save_mesh(tri_mesh_name, bm)
            bm.free()

            # bpy.ops.object.modifier_add(type='SUBSURF')
            # bpy.ops.object.shade_smooth()

            bpy.ops.object.mode_set(mode='EDIT')

            scene = bpy.context.scene
            opt_names = [sub.name for sub in option_classes]
            current_names = [opt.name for opt in scene.options]

            for cur_name in current_names:
                if cur_name not in opt_names:
                    ind = scene.options.find(cur_name)
                    scene.options.remove(ind)
            for i, opt in enumerate(option_classes):
                if opt.name not in current_names:
                    option = scene.options.add()
                    option.name = opt.name
                # Move to correct position
                ind = scene.options.find(opt.name)
                scene.options[ind].disp_name = opt.disp_name
                scene.options.move(ind, i)

            model.current_params = [p.primal for p in model.symbols]
            Opt.interpret()

        return {'FINISHED'}

class Module:
  active_modules = {}

  @classmethod
  def add_module(cls, name):
    if name in cls.active_modules:
      cls.active_modules[name].free()
    cls.active_modules[name] = crun.DyC(name)
    return cls.active_modules[name]

  def __init__(self, name):
    self.name = name
    self.compiled = False
    self.active_module = None
    self.heads = []
    self.bodies = []


  def add_function(self, head, body):
    if self.compiled:
      raise Exception("Cannot add functions to previously compiled modules")
    self.heads.append(head)
    self.bodies.append(body)

  def compile(self):
    if self.compiled:
      raise Exception("Compiling previously compiled module")
    crun.compiler(self.name,"".join(self.heads),"".join(self.bodies))
    self.active_module = Module.add_module(self.name)
    self.compiled = True

  def load_function(self, len_inputs, len_outputs, name):
    if not self.compiled:
      raise Exception("Module must be compiled before functions can be loaded")
    # TODO (PAQ): Check that the module has not been freed
    return self.active_module.load_function(len_inputs, len_outputs, name)

  def load_scalar_function(self, len_inputs, name):
    if not self.compiled:
      raise Exception("Module must be compiled before functions can be loaded")
    return self.active_module.load_scalar_function(len_inputs, name)

class BIMODEL_OT_Compile(bpy.types.Operator):
  bl_idname = "bimodel.compile"
  bl_label = "Compile"
  bl_description = "Compile the generated traces"
  bl_options = {"REGISTER"}

  @classmethod
  def poll(cls, context):
      return Opt.is_interpreted()

  # We want to compile:
  # - The forward propagation of model -> vertices (for area, vol)
  # - Staged functions that will return the loss for our vertex loss, area loss, and volume loss.
  #   These require staging because we need appropriate selection information to compute the vertex loss.
  # - Staged functions that will return the _derivative_ for area & volume loss. Again, this requires
  #   selection information to compute the derivative of the vertex loss component.
  @profile
  def execute(self, context):
    from objectives import lib, ffi
    try:
      Opt.compile()

      return {"FINISHED"}
    except Exception as e:
      save_edit_mesh()
      raise e

def rounded(values):
    return [round(x, DECIMALS) for x in values]

def get_selected_sync():
    option = get_selected_option()
    if option and option.name in Opt.synced:
        return Opt.synced[option.name]

def get_selected_option():
    index = bpy.context.scene.option_index
    if index < 0 or index >= len(bpy.context.scene.options):
        return None
    return bpy.context.scene.options[index]

def update_com_mesh():
    if Opt.is_interpreted():
        data = edit_mesh_to_numpy()
        if data:
            V, F = data
            c = center_of_mass(V, F)
            # print("Center of mass", c)
            bm = bmesh.new()
            bm.verts.new(Vector(c))
            mat_loc = mathutils.Matrix.Translation(c)
            bmesh.ops.create_uvsphere(
                bm,
                u_segments=10,
                v_segments=10,
                matrix=mat_loc,
                diameter=0.2)
            save_mesh(com_mesh_name, bm)
            bm.free()

def view_selected_option():
    if Opt.is_synced() and get_selected_sync():
        sync, _ = get_selected_sync()
        if not math.isnan(sync['fun']):
            BiModel.last_prog.update_params(sync['x'])
            update_com_mesh()
            bpy.context.scene.params = f"{rounded(sync['x'].tolist())}"

class DrawHandler:

    point_batch = None
    option_batch = None

    @classmethod
    def create_batch(cls):
        bm = edit_mesh()
        if bm:
            coords = [bm.verts[v].co for v in Opt.selected_vertices]
            cls.point_batch = batch_for_shader(cls.point_shader, 'POINTS', {"pos": coords})

    @classmethod
    def create_batch_option(cls, coords):
        bm = edit_mesh()
        if bm:
            vecs = [Vector(v) for v in coords]
            indices = [(e.verts[0].index, e.verts[1].index) for e in bm.edges]
            cls.option_batch = batch_for_shader(cls.option_shader, 'LINES', {"pos": vecs}, indices=indices)

    @classmethod
    def draw(cls):
        if cls.point_batch:
            cls.point_shader.bind()
            cls.point_shader.uniform_float("color", (1, 0, 0, 1))
            bgl.glPointSize(10)
            cls.point_batch.draw(cls.point_shader)
        if cls.option_batch:
            cls.option_shader.bind()
            cls.option_shader.uniform_float("color", (1, 1, 0, 1))
            cls.option_batch.draw(cls.option_shader)


        # bm = edit_mesh()
        # if bm:
        #     coords = [bm.verts[v].co for v in Opt.selected_vertices]
        #     point_batch = batch_for_shader(cls.point_shader, 'POINTS', {"pos": coords})
        #     point_batch.draw(cls.point_shader)


    @classmethod
    def register_draw(cls):
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.draw, (), 'WINDOW', 'POST_VIEW')
        cls.point_shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        cls.option_shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        # cls.point_batch = batch_for_shader(cls.point_shader, 'POINTS', {"pos": []})
        print("Draw loaded")

    @classmethod
    def unregister_draw(cls):
        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, "WINDOW")

class BIMODEL_OT_Handle(bpy.types.Operator):
    bl_idname = "bimodel.handle"
    bl_label = "Handle"
    bl_description = "Update model handles"
    bl_options = {"REGISTER", "UNDO"}

    action: bpy.props.EnumProperty(
        items=(
            ('ADD', "ADD", ""),
            ('REMOVE', "REMOVE", ""),
            ('CLEAR', "CLEAR", "")
        )
    )

    @classmethod
    def poll(cls, context):
        return Opt.is_compiled()

    def execute(self, context):
        bm = edit_mesh()
        if self.action == 'ADD':
            Opt.selected_vertices.update(get_vert_selection(bm))
        elif self.action == 'REMOVE':
            for v in get_vert_selection(bm):
                Opt.selected_vertices.remove(v)
        else:
            Opt.selected_vertices.clear()
        print("Current selected:", Opt.selected_vertices)
        DrawHandler.create_batch()
        return {'FINISHED'}


class BIMODEL_OT_Sync(bpy.types.Operator):
    bl_idname = "bimodel.sync"
    bl_label = "Sync"
    bl_description = "Sync edits to program"
    bl_options = {"REGISTER", "UNDO"}

    sync_all: bpy.props.BoolProperty()

    @classmethod
    def poll(cls, context):
        return Opt.is_compiled()

    @profile
    def execute(self, context):
      global num_runs
      try:
        
        Opt.render_option = False
        bm = edit_mesh()
        # bm_trace = load_mesh(trace_mesh_name)
        bm_sync = load_mesh(trace_mesh_name)

        for i, vert in enumerate(bm.verts):
            if vert.index in Opt.selected_vertices:
                bm_sync.verts[i].co = vert.co
            else:
                vert.co = bm_sync.verts[i].co

        save_mesh(sync_mesh_name, bm_sync)
        bm_sync.free()
        # bm_trace.free()

        x0 = Opt.current_params.copy()
        Opt.sync()
        if self.sync_all:
            for option in bpy.context.scene.options:
                if option.sync or option.name == "Original":
                    Opt.sync_op(option.name, x0)
                    # Reload the UI after each optimization
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            context.scene.option_index = -1
        else:
            option = get_selected_option()
            if option:
                Opt.sync_op("Original", x0)
                Opt.sync_op(option.name, x0)

                view_selected_option()
        Opt.render_option = True
        return {"FINISHED"}
      except Exception as e:
        save_edit_mesh()
        Opt.render_option = True
        raise e

# Each constraint objective is evaluated separately, so we _have_ to give each
# one its own autodiff structure, so that it can propagate correctly. The vast
# majority of these will not be very deep, so they're small clones.
# TODO (PAQ): This is not used anymore but might be useful for combining with other nodes.
def constraint_objective(x0, ctx, original_roots, original_node):
  roots, node = ctx.clone(original_roots, original_node)
  ctx.clean(node)
  # ctx.update({ roots[i]: roots[i].primal for (i,_) in enumerate(x0)}, log=True)

  # print(f"Constraint: {b}")
  def constraint_function(parameters):
      ctx.update({ roots[i] : p  for (i,p) in enumerate(parameters) })
      result = node.primal
      # print(f"Inequality loss = {result} (P = {parameters})")
      return result
  def constraint_derivative(parameters):
      ctx.update({ roots[i] : p  for (i,p) in enumerate(parameters) })
      node.d(0.01)
      result = auto.DiffContext.arrayGradient(roots)
      # print(f"Inequality gradient = {result} (P = {parameters})")
      return result
  return constraint_function, constraint_derivative

# We don't need to calculate our loss dynamically. We stage the computation here,
# creating a root node whose value will be updated every time the model parameters are updated.
# To evaluate the actual loss function & derivative we will call its primal and gradient.
# We want a least-squares distance, so instead of taking the square root and then squaring it,
# we'll just add all of the positional differences.
#
# Include optional weights for each vertex
def vertex_loss(traces, selected, targets, weights=None) -> auto.Diff:
    if len(selected) == 0:
        return auto.Diff.zero()
    total_loss = []
    for vert in selected:
        diff = subVert(traces[vert], targets[vert])
        if weights is not None:
            diff = mulVert(diff, weights[vert])
        total_loss.append(lenVec(diff))
    return auto.Diff.sum(total_loss)

def vertex_loss_l1(traces, selected, targets, weights=None) -> auto.Diff:
    if len(selected) == 0:
        return auto.Diff.zero()
    total_loss = []
    for vert in selected:
        diff = subVert(traces[vert], targets[vert])
        if weights is not None:
            diff = mulVert(diff, weights[vert])
        total_loss.append(abs(diff[0]))
        total_loss.append(abs(diff[1]))
        total_loss.append(abs(diff[2]))
    return auto.Diff.sum(total_loss)

def edge_loss(traces, edge_info, weights=None):
    if len(edge_info) == 0:
        return auto.Diff.zero()

    total_loss = []
    for v1, v2, edge_length, _ in edge_info:
        d_edge = sqrt(distSquare(traces[v1], traces[v2])) - edge_length
        if weights is not None:
            factor = max(weights[v1], weights[v2])
            total_loss.append((factor * factor) * d_edge * d_edge)
        else:
            total_loss.append(d_edge * d_edge)
    return auto.Diff.sum(total_loss)

def edge_loss_l1(traces, edge_info, weights=None):
    if len(edge_info) == 0:
        return auto.Diff.zero()

    total_loss = []
    for v1, v2, edge_length, _ in edge_info:
        d_edge = sqrt(distSquare(traces[v1], traces[v2])) - edge_length
        if weights is not None:
            factor = max(weights[v1], weights[v2])
            total_loss.append(abs(factor * d_edge))
        else:
            total_loss.append(abs(d_edge))
    return auto.Diff.sum(total_loss)

def param_loss(x0, parameters):
  loss = 0.0
  for i, root in enumerate(parameters):
    dp = x0[i]-root
    loss += (dp*dp)
  return 2.0 * loss

def provenance_loss(x0, roots, traces, selected) -> auto.Diff:
  weights = auto.DiffContext.computeProvenanceWeights(selected, roots, traces)
  print(f"Found provenance weights: {weights}")
  def getWeight(weight, x0, root):
    diff = x0-root
    return weight * (diff*diff)
  loss = sum([ getWeight(weights[i],x0[i],roots[i]) for i in range(len(roots)) ])
  return 2.0 * loss

# Objective is a function that takes:
# x : a 1D array of the current model parameters
# *args, a tuple of the rest of the stuff passed through the args parameter
# given to optimize.minimize. In our case thse are:
#      ctx: our autodiff context (and cache)
#    roots: the symbolic variables that are our parameters
# lossNode: the scalar node returned by the loss function as described above.
def objective(x, ctx, roots, lossNode: auto.Diff) -> float:
  # We want some relatively efficient way of evaluating the computation graph we've constructed
  # and stored in `model`. To do this we have `assignments` and `traces` as fields of `model`.
  # What we will do is:
  # -> Batch update the assignments
  # -> Have these assignments propagate the new primal values down through the computation tree,
  #    in a forward pass. Memoize any update where the value is basically the same, and don't
  #    propagate that. At this point, we can extract the vertex primals for the vertices we are
  #    interested in.

  ctx.update({ roots[i] : p  for (i,p) in enumerate(x) })
  # print(f"Evaluated loss ({lossNode.primal}) P = ({x})")
  return lossNode.primal

# Likewise our der_objective is the jacobian of the loss function w/r/t each
# parameters, with the same arguments. This should return a vector of the
# gradients for each parameter in the order they are supplied.
def der_objective(x, ctx, roots, lossNode: auto.Diff):
    # To actually evaluate the derivative using autodiff, we will need to do all of the
    # above (again, checking if the parameters have changed -- if they haven't, we will
    # reuse the cached value from the objective evaluation and _just call .d() on it_.)
    # To make sure we get all of the parameters, we fetch the gradient value from each root.
    ctx.update({ roots[i] : p for (i,p) in enumerate(x)}) # Use the "current" parameters
    lossNode.d(0.10)
    result = auto.DiffContext.arrayGradient(roots)
    # print(f"Evaluated derivative ({result}) P = ({x})")
    return result

# The problem with this is that it is, in fact, just plain too deep for python's recursion to handle.
# We will need to resort to just using numpy to perform the computation, extracting the current positions
# from the vertex primals as needed. The following function should not be called -- it will not only take
# quite a while to compute, it will also overflow the stack immediately on calling .d().
#
# N = number of vertices in the model, cubic # of ops in that. For our otherwise
# reasonable 500-vertex model, 125M+ operations becomes too much for autodiff to handle at
# a single-element level of granularity. (Which, honestly, is reasonable).
def biharmonic_loss_node(x, ctx, roots, traces, initial, Q, editLossNode) -> float:
    ctx.update({ roots[i] : p for (i,p) in enumerate(x)})
    # all vertex positions (current)  -  all vertex positions initially (at start)
    D = np.array([subVec(valueOf(traces[vert]),initial[vert]) for vert in range(len(initial))])
    Dt = np.transpose(D)
    DtQD = Dt @ Q @ D
    weight = 1000.0
    E = np.trace(DtQD)    # get the trace of the matrix (diagonal sum)

    # Frob norm of our edit distance vector is the sqrt of the L2 in this case.
    frob_norm = sqrt(editLossNode.primal)
    loss = E + weight * frob_norm
    return loss

def quadratic_objective(x, traces, derivs, A, B, b, bc, weight):
    Z = traces(x)
    Zt = np.transpose(Z)
    ZtAZ = 0.5 * (Zt @ A @ Z)
    ZtB = Zt @ B
    ZtAZ_ZtB = ZtAZ + ZtB
    E = -1 * np.trace(ZtAZ_ZtB)

    handle_constraints = Z[b] - bc
    frob_norm = np.linalg.norm(handle_constraints, ord='fro')**2
    loss = E + weight * frob_norm
    return loss

def quadratic_objective_der(x, traces, derivs, A, B, b, bc, weight):
    D = traces(x)
    Dt = np.transpose(D)
    At = np.transpose(A)
    Bt = np.transpose(B)
    dD = derivs(x)

    U = D[b] - bc

    e = 1e-7
    eps = e * np.eye(len(x))
    fx = []
    fx2 = []
    for xi in range(len(x)):
        dDi = dD[xi, :, :]

        DtAtBdDi = ((0.5 * (Dt @ At)) + Bt) @ dDi

        DhUht = weight * np.transpose(U)
        dDh = dDi[b]

        DhUht_dDh = DhUht @ dDh

        DtAtBdDi_DhUht_dDh = DtAtBdDi + DhUht_dDh
        fxi = 2 * np.trace(DtAtBdDi_DhUht_dDh)

        # fxi = 2 * np.trace(DhUht_dDh)
        fx.append(fxi)

        # Finite difference
        x1 = x - eps[xi]
        x2 = x + eps[xi]
        f1 = quadratic_objective(x1, traces, derivs, A, B, b, bc, weight)
        f2 = quadratic_objective(x2, traces, derivs, A, B, b, bc, weight)
        fxi2 = (f2 - f1) / (2.0 * e)
        fx2.append(fxi2)

    ret_fx = np.array(fx2)
    return ret_fx

def compile_objectives():
    with open("objectives_source.c","r") as f:
        bodies = f.read()

    headers = [
        "double surface_area(double*, int*, int);",
        "double volume(double*, int *, int);",
        "double edit(double*, double*, int*, int);",
        "void center_of_mass(double*, int*, int, double*);"
    ]

    crun.compiler("objectives", "\n".join(headers), bodies)


class Opt:
    try:
        ob = crun.DyC("objectives")
    except Exception as ex:
        print("Objectives not found, generating objectives")
        compile_objectives()
        ob = crun.DyC("objectives")

    options = {}
    synced = {}

    render_option = True

    current_params = None
    last_check_params = None
    last_stage = 0

    selected_vertices = set()

    trace_graph_size = None

    @classmethod
    def reset(cls):
        cls.last_stage = 0
        cls.current_params = None
        cls.last_check_params = None
        cls.options.clear()
        cls.synced.clear()
        cls.selected_vertices.clear()

    @classmethod
    def update_params(cls, params):
        cls.current_params = params.copy()

    @classmethod
    def is_interpreted(cls):
        return cls.last_stage >= 1

    @classmethod
    def is_compiled(cls):
        return cls.last_stage >= 2

    @classmethod
    def is_synced(cls):
        return cls.last_stage >= 3

    @classmethod
    def interpret(cls):
        cls.reset()
        cls.last_stage = 1
        cls.update_params(BiModel.last_prog.current_params)

        cls.options.clear()
        for i, opt in enumerate(option_classes):
            print("Creating", opt.name)
            o = opt()
            o.index = i
            cls.options[opt.name] = o


    @classmethod
    def compile(cls):
        assert cls.is_interpreted()

        from objectives import ffi, lib

      
        bm = edit_mesh()

        m = Module('model')
        cls.m = m
        m.add_function(*auto.Diff.support_functions())

        # JIT ourselves a forward pass, from roots -> trace values
        ctx = auto.DiffContext()
        model = BiModel.last_prog

        # Clone ourselves a copy o the diff graph, both so that we can serialize it,
        # but so that we can use it in conjunction with the vol. objective when deriving stuff.
        original_roots  = model.symbols
        original_traces = [model.traces[i] for i in range(len(model.traces))]
        flat_traces = flatten(original_traces)
        roots, traces = ctx.clone(original_roots, flat_traces)
        traces = chunks(traces,3)

        cls.trace_graph_size = auto.Diff.calc_size(flat_traces)
        print("Graph size", cls.trace_graph_size)
        m.add_function(*ctx.serialize_forward_traces(roots, traces))

        # JIT ourselves a backwards one too, computing volume derivative
        V, F = mesh_to_numpy(tri_mesh_name)
 
        for name, op in cls.options.items():
            op.add_functions(m, ctx, roots, traces, V, F)
       
        m.add_function(*ctx.serialize_forward_jacobian(roots, flatten(traces)))

        ineq_roots, ineq_traces = ctx.clone(model.symbols, model.inequality_constraints)
        m.add_function(*ctx.serialize_forward_array(ineq_roots, ineq_traces, tag="_constraint"))
        m.add_function(*ctx.serialize_forward_jacobian(ineq_roots, ineq_traces, tag="_constraint"))

        edge_nodes = []
        for edge in bm.edges:
            v1 = traces[edge.verts[0].index]
            v2 = traces[edge.verts[1].index]
            edge_nodes.append(distSquare(v1, v2))

        edge_roots, edge_traces = ctx.clone(model.symbols, edge_nodes)
        m.add_function(*ctx.serialize_forward_array(edge_roots, edge_traces, tag="_edge"))
        m.add_function(*ctx.serialize_forward_jacobian(edge_roots, edge_traces, tag="_edge"))

        print("Beginning compile...")
        m.compile()

        # Okay, it's compiled. Let's load up the functions. These are:
        # Input (= len(roots)), Output (= 3*len(traces)), Gradient (= len(roots))
        num_roots = len(roots)
        num_traces = len(traces)
        num_trace_array = num_traces * 3
        forward, _ = m.load_function(num_roots, num_trace_array, "forward_array")
        forward_jacobian, _ = m.load_function(num_roots, num_trace_array * num_roots, "forward_jacobian")

        num_constraints = len(model.inequality_constraints)
        forward_constraints, _ = m.load_function(num_roots, num_constraints, "forward_array_constraint")
        forward_jacobian_constriants, _ = m.load_function(num_roots, num_constraints * num_roots, "forward_jacobian_constraint")

        num_edges = len(bm.edges)
        forward_edges, _ = m.load_function(num_roots, num_edges, "forward_array_edge")
        forward_jacobian_edges, _ = m.load_function(num_roots, num_edges * num_roots, "forward_jacobian_edge")

        for name, op in cls.options.items():
            op.load_functions(m, num_roots)

        # Allocate & load up the flattened face index array.
        bm_tri = load_mesh(tri_mesh_name)
        faceIndices = bmesh_face_indices(bm_tri)
        faces = ffi.new("int[]",len(faceIndices))
        faces[0:len(faceIndices)] = faceIndices


        # Directly call F
        def trace_vertices(parameters):
            _, vals = forward(parameters)
            return vals.reshape(num_traces, 3)

        def trace_jacobian(parameters):
            _, vals = forward_jacobian(parameters)
            return vals.reshape(num_roots, num_traces, 3)

        def constraints(parameters):
            _, vals = forward_constraints(parameters)
            return vals

        def constraints_jacobian(parameters):
            _, vals = forward_jacobian_constriants(parameters)
            # TODO (PAQ): Might be a better way to do this
            return np.transpose(vals.reshape(num_roots, num_constraints))

        def trace_edges(parameters):
            _, vals = forward_edges(parameters)
            return vals

        def trace_jacobian_edges(parameters):
            _, vals = forward_jacobian_edges(parameters)
            return vals.reshape(num_roots, -1)


        cls.trace_vertices = trace_vertices
        cls.trace_jacobian = trace_jacobian
        cls.constraints = constraints
        cls.constraints_jacobian = constraints_jacobian

        cls.trace_edges = trace_edges
        cls.trace_jacobian_edges = trace_jacobian_edges

        # TODO (PAQ): Should this jacobian be scaled by 0.001?
        cls.constraints = {'type': 'ineq', 'fun' :  cls.constraints, 'jac' :  cls.constraints_jacobian}

        cls.last_stage = 2

    @classmethod
    def sync_op(cls, name, x0):
        cls.synced[name] = cls.options[name].optimize(x0)

    @classmethod
    def setup_sync(cls):
        assert cls.is_compiled()

        bm = edit_mesh()

        cls.ctx = auto.DiffContext()

        model = BiModel.last_prog
        # TODO: Do the roots and traces need to be created each sync?
        trace_nodes = [node for i in range(len(model.traces)) for node in model.traces[i]]
        cls.roots, flat_traces = cls.ctx.clone(model.symbols, trace_nodes)
        cls.traces = chunks(flat_traces, 3)


        cls.V0, cls.F = mesh_to_numpy(tri_mesh_name)
        cls.V = verts_to_numpy(bm)
        cls.targets = [vert.co for vert in bm.verts] # TODO: replace everywhere with V

        cls.Area0 = squared_surface_area_float(cls.V0, cls.F) # TODO: we could use lib.area()
        cls.Vol0 = volume(cls.V0, cls.F) # TODO: we could use lib.volume() for this
        cls.COM0 = center_of_mass(cls.V0, cls.F)

        cls.selected_verts = list(cls.selected_vertices)
        cls.unselected_verts = [vert.index for vert in bm.verts if vert.index not in cls.selected_vertices]

        # Use igl to compute geodesic to each vertice
        # Selected verts will have a distance of 0

        edge_info = [(edge.verts[0].index, edge.verts[1].index, edge.calc_length() * edge.calc_length(), edge.index) for edge in bm.edges]

        cls.selected_edges = [e for e in edge_info if bm.verts[e[0]].select and bm.verts[e[1]].select]
        cls.half_selected_edges = [e for e in edge_info if not (bm.verts[e[0]].select == bm.verts[e[1]].select)]
        cls.unselected_edges = [e for e in edge_info if not (bm.verts[e[0]].select or bm.verts[e[1]].select)]

        if LOCALIZATION:
            VV = cls.V0.copy()
            FF = cls.F.copy()
            SRC = np.array(list(cls.selected_verts),dtype=np.int)
            DST = np.arange(len(cls.V0),dtype=np.int)
            dist = igl.exact_geodesic(VV, FF, SRC, DST)

            factor = dist.sum()
            if factor > 0:
                dist = dist / factor
            cls.geodesics = dist

            cls.edge_geodesics = np.array([max(cls.geodesics[v1], cls.geodesics[v2]) for v1, v2, _, _ in edge_info])

        cls.E0 = np.array([edge.calc_length() * edge.calc_length() for edge in bm.edges])

        # Stage our loss so we don't need to evaluate it every time --
        # we will update the parameters and then just check what value this node has.
        cls.editLossNode = vertex_loss(cls.traces, cls.selected_verts, cls.targets)


    @classmethod
    def sync(cls):
        assert cls.is_compiled()
         # Make sure the edited mesh is written
        # save_edit_mesh()

        cls.synced.clear()


        cls.setup_sync()

        cls.last_stage = 3

    @classmethod
    def available(cls):
        return [sub.name for sub in option_classes]

    def add_functions(self, m, ctx, roots, traces, V, F):
        pass

    def load_functions(self, m, num_roots):
        pass

    def optimize(self, x0, node):
        Opt.ctx.clean(node)
        return self.run_optimize(objective, der_objective, x0, Opt.ctx, Opt.roots, node)

    def run_optimize(self, fun, jac, x0, *args):
        timer = perf_counter_ns()
        opt = sp.optimize.minimize(fun, x0,
                    method='SLSQP',
                    constraints=Opt.constraints,
                    jac=jac,
                    options={'ftol': 1e-9,'disp': False, 'maxiter': 100},
                    args=args)
        time = perf_counter_ns() - timer
        return {'x': opt.x, 'fun': opt.fun, 'iters': opt.nit}, time

    def weight(self):
        return bpy.context.scene.options[self.index].weight

@bpy.app.handlers.persistent
def handle_load(context):
    Opt.reset()
    if context:
        context.scene.select_names = ""
        context.scene.params = ""

def select_string():
    prog = BiModel.last_prog
    if hasattr(prog, 'vert_objects'):
        bm = edit_mesh()

        verts = get_vert_selection(bm)
        objects = {}

        for vert in verts:
            if vert not in prog.vert_objects:
                return f"Vert {vert} has no name"

            name, index = prog.vert_objects[vert]
            if name not in objects:
                objects[name] = []
            objects[name].append(index)
        return "+".join(map(lambda x: f"{ArrayNames.remake_sub(x[0])}.v({x[1]})", objects.items()))
    return "---"

updates = 0
@bpy.app.handlers.persistent
def handle_update(scene, depsgraph):
    global updates
    updates += 1

    model_update = False
    for update in depsgraph.updates:
        if update.id.original == bpy.context.active_object and update.is_updated_geometry and not update.is_updated_transform:
            model_update = True

    # print("Handle update", updates, model_update)
    
    if model_update and Opt.is_interpreted() and BiModel.has_model():
        if edit_mesh():
            scene.select_names = select_string()
        update_com_mesh()

        option = get_selected_option()
        if Opt.is_compiled() and Opt.render_option and len(Opt.selected_vertices) > 0 and scene.presync:
            Opt.render_option = False
            Opt.setup_sync()
            if Opt.last_check_params is not None:
                x0 = Opt.last_check_params
            else:
                x0 = Opt.current_params.copy()
            sync, time = Opt.options[option.name].optimize(x0)
            print("Opt time:", time/1e9, "iers:", sync['iters'])
            if not math.isnan(sync['fun']):
                x = sync['x']
                coords =  Opt.trace_vertices(x)
                DrawHandler.create_batch_option(coords)
                Opt.last_check_params = x
            Opt.render_option = True
        else:
            DrawHandler.option_batch = None
        DrawHandler.create_batch()


def norm_l2(f, x, f0, inds, weight=None):
    fx = f(x)
    diff = (fx-f0)[inds]
    if weight is not None:
        diff = diff * weight[inds]
    return np.linalg.norm(diff)**2

def norm_l2_derivative(f, df, x, f0, inds, weight=None):
    fx = f(x)
    jacobian = df(x)
    var_count = jacobian.shape[0]
    derivative = np.zeros(var_count)
    for xi in range(len(derivative)):
        partial = jacobian[xi]

        diff = (fx - f0)[inds]
        if weight is not None:
            diff = diff * weight[inds] * weight[inds]

        diff = np.transpose(diff)

        der = diff @ (partial[inds])
        if isinstance(der, float):
            fxi = 2 * der
        else:
            fxi = 2 * np.trace(der)

        derivative[xi] = fxi
    return derivative

def norm_l1(f, x, f0, inds, weight=None):
    fx = f(x)
    diff = (fx - f0)[inds]
    if weight is not None:
        diff = diff * weight[inds]
    return np.sum(np.abs(diff))

def norm_l1_derivative(f, df, x, f0, inds, weight=None):
    fx = f(x)
    jacobian = df(x)
    var_count = jacobian.shape[0]
    derivative = np.zeros(var_count)
    for xi in range(len(derivative)):
        partial = jacobian[xi]

        diff = (fx - f0)[inds]

        s = np.sign(diff)

        if weight is not None:
            dfxi = np.sum(s * partial[inds] * weight[inds])
        else:
            dfxi = np.sum(s * partial[inds])


        derivative[xi] = dfxi

    return derivative

class Original(Opt):
    name = "Original"
    disp_name = "Original"

    def optimize(self, x0):
        def loss_fun(params):
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return edit_loss
        loss1 = loss_fun(x0)
        return {'x': np.array(x0), 'fun': loss1, 'iters':0}, 0

class EditComp(Opt):
    name = "Edit Comp"
    disp_name = "Edit"

    def optimize(self, x0):
        def loss(params):
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return edit_loss
        def der(params):
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * edit_der

        return self.run_optimize(loss, der, x0)

class VertexComp(Opt):
    name = "Vertex Comp"
    disp_name = "Vertex L2"

    def optimize(self, x0):
        def loss(params):
            vertex_loss = norm_l2(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts)
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return vertex_loss + self.weight() * edit_loss

        def der(params):
            vertex_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts)
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (vertex_der + self.weight() * edit_der)

        return self.run_optimize(loss, der, x0)

class VertexL1Comp(Opt):
    name = "Vertex L1 Comp"
    disp_name = "Vertex"

    def optimize(self, x0):
        def loss(params):
            vertex_loss = norm_l1(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts)
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return vertex_loss + self.weight() * edit_loss

        def der(params):
            vertex_der = norm_l1_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts)
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (vertex_der + self.weight() * edit_der)

        return self.run_optimize(loss, der, x0)

class VertexLocalComp(Opt):
    name = "Vertex Localization Comp"
    disp_name = "Localized Vertex"

    def optimize(self, x0):
        def loss(params):
            vertex_loss = norm_l2(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, Opt.geodesics[:, np.newaxis])
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return vertex_loss + self.weight() * edit_loss

        def der(params):
            vertex_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, Opt.geodesics[:, np.newaxis])
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (vertex_der + self.weight() * edit_der)

        return self.run_optimize(loss, der, x0)

class VertexLocalL1Comp(Opt):
    name = "Vertex Localization L1 Comp"
    disp_name = "Localized Vertex"

    def optimize(self, x0):
        def loss(params):
            vertex_loss = norm_l1(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, Opt.geodesics[:, np.newaxis])
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return vertex_loss + self.weight() * edit_loss

        def der(params):
            vertex_der = norm_l1_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, Opt.geodesics[:, np.newaxis])
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (vertex_der + self.weight() * edit_der)

        return self.run_optimize(loss, der, x0)

class EdgeComp(Opt):
    name = "Edge Comp"
    disp_name = "Edge"

    def optimize(self, x0):
        edge_inds = [ind for _, _, _, ind in Opt.unselected_edges]
        def loss(params):
            edge_loss = norm_l2(Opt.trace_edges, params, Opt.E0, edge_inds)
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return edge_loss + self.weight() * edit_loss

        def der(params):
            edge_der = norm_l2_derivative(Opt.trace_edges, Opt.trace_jacobian_edges, params, Opt.E0, edge_inds)
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (edge_der + self.weight() * edit_der)
        return self.run_optimize(loss, der, x0)

class EdgeL1Comp(Opt):
    name = "Edge L1 Comp"
    disp_name = "Edge"

    def optimize(self, x0):
        edge_inds = [ind for _, _, _, ind in Opt.unselected_edges]
        def loss(params):
            edge_loss = norm_l1(Opt.trace_edges, params, Opt.E0, edge_inds)
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return edge_loss + self.weight() * edit_loss

        def der(params):
            edge_der = norm_l1_derivative(Opt.trace_edges, Opt.trace_jacobian_edges, params, Opt.E0, edge_inds)
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            return 0.1 * (edge_der + self.weight() * edit_der)
        return self.run_optimize(loss, der, x0)

class EdgeLocalComp(Opt):
    name = "Edge Localization Comp"
    disp_name = "Localized Edges"

    def optimize(self, x0):
        edge_inds = [ind for _, _, _, ind in Opt.unselected_edges]
        def loss(params):
            edge_loss = norm_l2(Opt.trace_edges, params, Opt.E0, edge_inds, weight=Opt.edge_geodesics)
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            return edge_loss + self.weight() * edit_loss
        norm_l2(Opt.trace_edges, [2], Opt.E0, edge_inds, weight=Opt.edge_geodesics)
        return self.run_optimize(loss, '2-point', x0)

class Param(Opt):
    name = "Parameter"
    disp_name = "Parameter"
    def optimize(self, x0):
        paramLossNode = param_loss(x0, Opt.roots) + self.weight() * Opt.editLossNode
        return super().optimize(x0, paramLossNode)

class Provenance(Opt):
    name = "Provenance"
    disp_name = "Provenance"

    def optimize(self, x0):
        provLossNode = provenance_loss(x0, Opt.roots, Opt.traces, Opt.selected_verts) + self.weight() * Opt.editLossNode
        return super().optimize(x0, provLossNode)

class Area2(Opt):
    name = "Area 2"
    disp_name = "Area"

    @staticmethod
    def area_objective_diff(traces, tri_F, Area0):
            def areaDiff(tri_V, tri_F):
                total_area = 0.0
                for tri in tri_F:
                    v1 = tri_V[tri[0]]
                    v2 = tri_V[tri[1]]
                    v3 = tri_V[tri[2]]
                    a = distSquare(v1, v2)
                    b = distSquare(v1, v3)
                    c = distSquare(v2, v3)
                    A = (2.0*a*b + 2.0*b*c + 2.0*c*a - a*a - b*b - c*c) / 16.0
                    total_area += A
                return total_area
            Area = areaDiff(traces, tri_F) - Area0
            return Area * Area

    def add_functions(self, m, ctx, roots, traces, V, F):
        Area0 = squared_surface_area_float(V, F)
        Area0Var = diffVar("area0_0", diffConst(Area0))
        aLossVar = Area2.area_objective_diff(traces, F, Area0Var)
        ctx.clean(aLossVar)

        m.add_function(*ctx.serialize_forward(roots+[Area0Var], aLossVar,tag="_area2"))
        m.add_function(*ctx.serialize_reverse(roots+[Area0Var], aLossVar,step=1.0,tag="_area2"))

    def load_functions(self, m, num_roots):
        self.area_loss_func = m.load_scalar_function(num_roots+1, "forward_area2")
        self.area_der_func, _ = m.load_function(num_roots+1, num_roots+1, "reverse_derivative_area2")

    def optimize(self, x0):
        AREA_WEIGHT = 1.0 / 10.0
            
        def area_loss_func(params):
            return self.area_loss_func(np.concatenate((params, np.array([Opt.Area0]))))

        def area_der_func(params):
            return self.area_der_func(np.concatenate((params, np.array([Opt.Area0]))))[1][:len(params)]

        def loss(params):
            vertex_loss = norm_l2(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            area_loss = area_loss_func(params)
            return AREA_WEIGHT * area_loss + (self.weight() * edit_loss) + 0.01 * vertex_loss

        def der(params):
            vertex_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            area_der = area_der_func(params)
            return AREA_WEIGHT * area_der + (self.weight() * edit_der) + 0.01 * vertex_der

        return self.run_optimize(loss, der, x0)


class Volume2(Opt):
    name = "Volume2"
    disp_name = "Volume"

    @staticmethod
    def volume_objective_diff(traces, tri_F, V0):
        def volumeDiff(tri_V, tri_F):
            total_vol = 0.0
            for tri in tri_F:
                v1 = tri_V[tri[0]]
                v2 = tri_V[tri[1]]
                v3 = tri_V[tri[2]]
                total_vol += auto.Volume([v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2]])
            return total_vol
        V = volumeDiff(traces, tri_F) - V0
        return (V*V)

    def add_functions(self, m, ctx, roots, traces, V, F):
        Vol0 = volume(V, F)
        V0Var = diffVar("vol0_0", diffConst(Vol0))
        vLossVar = Volume2.volume_objective_diff(traces, F, V0Var)
        ctx.clean(vLossVar)

        m.add_function(*ctx.serialize_forward(roots+[V0Var], vLossVar,tag="_vol2"))
        m.add_function(*ctx.serialize_reverse(roots+[V0Var], vLossVar,step=1.0,tag="_vol2"))

    def load_functions(self, m, num_roots):
        self.vol_loss_func = m.load_scalar_function(num_roots+1, "forward_vol2")
        self.vol_der_func, _ = m.load_function(num_roots+1, num_roots+1, "reverse_derivative_vol2")
        
    def optimize(self, x0):
        VOLUME_WEIGHT = (1.0 / 50.0)
        def vol_loss_func(params):
            return self.vol_loss_func(np.concatenate((params, np.array([Opt.Vol0]))))

        def vol_der_func(params):
            return self.vol_der_func(np.concatenate((params, np.array([Opt.Vol0]))))[1][:len(params)]

        def loss(params):
            vertex_loss = norm_l2(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            vol_loss = vol_loss_func(params)
            return VOLUME_WEIGHT * vol_loss + (self.weight() * edit_loss) + 0.01 * vertex_loss

        def der(params):
            vertex_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            vol_der = vol_der_func(params)
            return VOLUME_WEIGHT * vol_der + (self.weight() * edit_der) + 0.01 * vertex_der

        return self.run_optimize(loss, der, x0)

def diff_finite(f, params, index, eps):
    delta = eps*np.eye(len(params))[index]
    return (f(params+delta)-f(params-delta))/(2*eps)

class COM2(Opt):
    name = "Center of Mass 2"
    disp_name = "Center of Mass"

    @staticmethod
    def com_objective_diff(traces, tri_F, COM0):
        def comDiff(tri_V, tri_F):
            total_vol = 0.0
            total_center = [0, 0, 0]
            for tri in tri_F:
                v1 = tri_V[tri[0]]
                v2 = tri_V[tri[1]]
                v3 = tri_V[tri[2]]
                V = auto.Volume([v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2]])
                total_vol += V
                center = mulVert(addVec(addVec(v1, v2), v3), 0.25 * V)
                total_center = addVert(center, total_center)
            return divVert(total_center, total_vol)
        COM = comDiff(traces, tri_F)
        return distSquare(COM, COM0)

    def add_functions(self, m, ctx, roots, traces, V, F):
        COM0 = center_of_mass(V, F)
        COM0Var = list(map(lambda x: diffVar(f"com_{x[0]}", diffConst(x[1])), enumerate(COM0.tolist())))
        comLossVar = COM2.com_objective_diff(traces, F, COM0Var)
        ctx.clean(comLossVar)

        m.add_function(*ctx.serialize_forward(roots+COM0Var, comLossVar,tag="_com2"))
        m.add_function(*ctx.serialize_reverse(roots+COM0Var, comLossVar,step=1.0,tag="_com2"))

    def load_functions(self, m, num_roots):
        self.com_loss_func = m.load_scalar_function(num_roots+3, "forward_com2")
        self.com_der_func, _ = m.load_function(num_roots+3, num_roots+3, "reverse_derivative_com2")


    def optimize(self, x0):
        def com_loss_func(params):
            return self.com_loss_func(np.concatenate((params, Opt.COM0)))

        def com_der_func(params):
            return self.com_der_func(np.concatenate((params, Opt.COM0)))[1][:len(params)]

        def loss(params):
            vertex_loss = norm_l2(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            vertex_loss_l1 = norm_l1(Opt.trace_vertices, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_loss = norm_l2(Opt.trace_vertices, params, Opt.V, Opt.selected_verts)
            com_loss = com_loss_func(params)
            return com_loss + (self.weight() * edit_loss) + 0.1 * vertex_loss_l1

        def der(params):
            vertex_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            vertex_der_l1 = norm_l1_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V0, Opt.unselected_verts, weight=Opt.geodesics[:, np.newaxis])
            edit_der = norm_l2_derivative(Opt.trace_vertices, Opt.trace_jacobian, params, Opt.V, Opt.selected_verts)
            com_der = com_der_func(params)
            return com_der + (self.weight() * edit_der) + 0.1 * vertex_der_l1


        return self.run_optimize(loss, der, x0)


class Biharmonic(Opt):
    name = "Biharmonic"
    disp_name = "Bi-harmonic energy"

    def optimize(self, x0):
        def biharmonic_loss(x, V0, traces, derivs, Q, verts_locations, handle_verts, weight):
            V = traces(x)
            D = V - V0
            Dt = np.transpose(D)
            DtQD = Dt @ Q @ D
            E = np.trace(DtQD)

            # Selected vertices
            handle_constraints = (V[handle_verts] - verts_locations[handle_verts])
            frob_norm = np.linalg.norm(handle_constraints, ord='fro')**2 # ||Dh(x)-Uh||^2

            loss = E + weight * frob_norm
            # print("Energy", E, "norm", frob_norm)
            return loss

        def biharmonic_loss_der(x, V0, traces, derivs, Q, verts_locations, handle_verts, weight):

            # df(x)/dxi = 2tr[D^T(x)Q^T(dD(x)/dxi) + (Dh(x)-Uh)^T*(dDh(x)/dxi)]

            V = traces(x)
            D = V-V0
            Dt = np.transpose(D)
            Qt = np.transpose(Q)
            dD = derivs(x)

            U = verts_locations - V0

            fx = []
            for xi in range(len(x)):
                dDi = dD[xi, :, :]

                DtQtdDi = Dt @ Qt @ dDi

                # This is equivalent to mytrace(x)[handle_verts] - verts_locations[handle_verts]
                DhUht = weight * np.transpose(D[handle_verts] - U[handle_verts])
                dDh = dDi[handle_verts]

                DhUht_dDh = DhUht @ dDh

                DtQtdDi_DhUht_dDh = DtQtdDi + DhUht_dDh
                fxi = 2 * np.trace(DtQtdDi_DhUht_dDh)
                fx.append(fxi)

            ret_fx = np.array(fx)
            return ret_fx
        Q = igl.harmonic_weights_integrated(Opt.V0, Opt.F, 2)
        return self.run_optimize(biharmonic_loss, biharmonic_loss_der, x0, Opt.V0, Opt.trace_vertices, Opt.trace_jacobian, Q, Opt.V, Opt.selected_verts, self.weight())

class ARAP(Opt):
    name = "ARAP"
    disp_name = "ARAP energy"

    def optimize(self, x0):
        K = precompute_K(Opt.V0, Opt.F, Opt.selected_verts)
        L = igl.cotmatrix(Opt.V0, Opt.F)
        n = len(Opt.traces)
        b = Opt.selected_verts
        bc = Opt.V[Opt.selected_verts]

        cur_x = x0
        iters = bpy.context.active_object.arap_iters
        total_time = 0
        total_iters = 0
        for _ in range(iters):
            U = Opt.trace_vertices(cur_x)
            weighted_cov = K.transpose() @ U
            R = np.zeros((3 * n, 3))

            for i in range(n):
                cov_i = weighted_cov[3*i:3*i+3, :]
                u, S, Vt = np.linalg.svd(cov_i)
                Ri = u @ Vt
                R[3*i:3*i+3, :] = Ri

            B = K @ R

            sync, time = self.run_optimize(quadratic_objective, quadratic_objective_der, cur_x, Opt.trace_vertices, Opt.trace_jacobian, L, B, b, bc, self.weight())
            total_time += time
            total_iters += sync['iters']
            cur_x, loss = sync['x'], sync['fun']
        return {'x': cur_x, 'fun': loss, 'iters': total_iters}, total_time

option_classes = []
option_classes.append(Original)
option_classes.append(EditComp)
option_classes.append(VertexL1Comp)
if LOCALIZATION:
    option_classes.append(VertexLocalL1Comp)
option_classes.append(EdgeL1Comp)
if LOCALIZATION:
    option_classes.append(EdgeLocalComp)
option_classes.append(Param)
if LOCALIZATION:
    option_classes.append(Provenance)
if LOCALIZATION:
    option_classes.append(Area2)
    option_classes.append(Volume2)
    option_classes.append(COM2)
    option_classes.append(ARAP)

option_classes.append(Biharmonic)

def render(fname, color, render_edges = False, cubes = False):
    scn = bpy.context.scene
    scn.render.resolution_x = 600
    scn.render.resolution_y = 800
    edge_color = (0, 0, 0)
    rgb = colors[color]
    rgb = rgb + (1,)
    bm = edit_mesh()

    # Assign material to object
    obj = bpy.context.active_object
    bpy.context.view_layer.objects.active = obj
    obj.data.materials.clear()
    bmat = bpy.data.materials['mat' + color]
    obj.data.materials.append(bmat)

    # hide all other objects
    for ob in bpy.context.view_layer.objects:
        if ob.name != obj.name:
            ob.hide_render = True
    obj.hide_render = False

    # Edge render settings
    if render_edges:
        scn.render.use_freestyle = True
        all_edges = [i for i in range(len(bm.edges))]
        set_edge_selection(bm, all_edges)
        bpy.ops.mesh.mark_freestyle_edge(clear=False)
        freestyle = bpy.context.view_layer.freestyle_settings
        for key in freestyle.linesets.keys():
            l = freestyle.linesets.get(key)
            freestyle.linesets.remove(l)
        LineSetV = freestyle.linesets.new('VisibleLineset')
        LineSetV.select_crease = False
        LineSetV.select_edge_mark = True
        #LineSetV.select_by_collection = True
        LineSetV.linestyle.thickness=3.0
        LineSetV.linestyle.color = edge_color
    else:
        bpy.context.scene.render.use_freestyle = False

    # Lighting
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.6, 1)

    # Camera
    # before running sync, set your prefered view and run bpy.ops.view3d.camera_to_view() or go to View->Align View -> Align Active Camera to View
    # create_tracking_cam(obj)

    # renderer settings
    bpy.data.scenes['Scene'].render.filter_size = 1.00
    bpy.data.scenes['Scene'].render.film_transparent = True
    bpy.data.scenes['Scene'].render.filepath = str(Path().absolute()) + "/" + fname
    bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer="", scene="")

def stitch_images(img_names):
    images = [Image.open(str(Path().absolute()) + "/" + x + ".png") for x in img_names]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for i, im in enumerate(images):
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    d = ImageDraw.Draw(new_im)
    x_offset = 0
    fnt = ImageFont.truetype('/System/Library/Fonts/Times.ttc', 50)
    for img_name in img_names:
    	d.text((x_offset+widths[0]/4, 30), img_name, font=fnt, fill=(255,255,255))
    	x_offset += im.size[0]

    new_im.save(str(Path().absolute())+"/"+'suggestions.png')

class BIMODEL_OT_PrintTraces(bpy.types.Operator):
    bl_idname = "bimodel.print_traces"
    bl_label = "Print Traces"
    bl_description = "Print traces of selected verts"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        bm = edit_mesh()
        verts = get_vert_selection(bm)
        prog = BiModel.last_prog
        trace = prog.traces[verts[0]]
        print(f"X:{trace[0].sy()}")
        print(f"Y:{trace[1].sy()}")
        print(f"Z:{trace[2].sy()}")
        
        return {"FINISHED"}

class BIMODEL_OT_CopyNames(bpy.types.Operator):
    bl_idname = "bimodel.copy_names"
    bl_label = "Print Names"
    bl_description = "Print names of selected verts"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        pyperclip.copy(context.scene.select_names)

        return {"FINISHED"}

class BIMODEL_OT_SelectEdit(bpy.types.Operator):
    bl_idname = "bimodel.select_edit"
    bl_label = "Select Edit"
    bl_description = "Select edited verts"
    bl_options = {"REGISTER"}


    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        bm = edit_mesh()
        bm_trace = load_mesh(trace_mesh_name)
        for i in range(len(bm.verts)):
            v1 = bm.verts[i].co
            v2 = bm_trace.verts[i].co
            bm.verts[i].select = (v1 - v2).length > 0.000001
        update_edit_mesh()
        return {"FINISHED"}

def get_open_text():
    for area in bpy.context.screen.areas:
        if area.type == 'TEXT_EDITOR':
            space = area.spaces.active
            return space.text



class BIMODEL_OT_SolveBiharmonic(bpy.types.Operator):
    bl_idname = "bimodel.biharmonic"
    bl_label = "Solve Bi-harmonid"
    bl_description = "Solve Bi-harmonic"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        bm = edit_mesh()
        verts = get_vert_selection(bm)

        if not verts:
            return {"FINISHED"}

        bm_def = load_mesh(def_mesh_name)
        bm_tri = load_mesh(tri_mesh_name)

        b = np.array([[i] for i in verts])
        bc = np.array([bm.verts[i].co - bm_tri.verts[i].co  for i in verts])

        V, F = mesh_to_numpy(tri_mesh_name)
        F = F.astype(dtype=np.int)

        w = igl.harmonic_weights(V, F, b, bc , 2)

        for i in range(len(bm_tri.verts)):
            bm_def.verts[i].co = bm_tri.verts[i].co + Vector(w[i])

        save_mesh(def_mesh_name, bm_def, hide=False)

        bm_def.free()
        bm_tri.free()

        return {"FINISHED"}

class BIMODEL_OT_SolveARAP(Operator):
    bl_idname = "bimodel.arap"
    bl_label = "Solve ARAP"
    bl_description = "Solve ARAP"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        bm = edit_mesh()
        verts = get_vert_selection(bm)

        if not verts:
            return {'FINISHED'}

        bm_def = load_mesh(def_mesh_name)
        bm_tri = load_mesh(tri_mesh_name)

        b = np.array([[i] for i in verts])
        bc = np.array([bm.verts[i].co for i in verts])

        V, F = mesh_to_numpy(tri_mesh_name)
        F = F.astype(dtype=np.int)

        n = len(bm_tri.verts)
        K = precompute_K(V, F, b)
        U = V
        L = igl.cotmatrix(V, F)
        iters = context.active_object.arap_iters
        prev_U = U
        for I in range(iters):
            U = solve_arap(prev_U, K, L, n, b, bc)
            prev_U = U

        for i in range(len(bm_tri.verts)):
            bm_def.verts[i].co = Vector(prev_U[i])

        save_mesh(def_mesh_name, bm_def, hide=False)
        bm_def.free()
        bm_tri.free()

        return {'FINISHED'}

def precompute_K(V, F, b):
    vert_count = V.shape[0]
    face_count = F.shape[0]

    # M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    # L = igl.cotmatrix(V, F) - np.eye(n)
    C = igl.cotmatrix_entries(V, F)

    K = np.zeros((vert_count, 3 * vert_count))
    # triples = []

    for f in range(face_count):

        for opposite_vertex in range(3):

            i = (opposite_vertex + 1) % 3
            j = (opposite_vertex + 2) % 3

            V_i = F[f, i]
            V_j = F[f, j]

            e_ij = (C[f, opposite_vertex] * (V[V_i, :] - V[V_j, :])) / 3.0

            for k in range(3):
                V_k = F[f, k]
                for beta in range(3):
                    K[V_i, 3 * V_k + beta] += e_ij[beta]
                    K[V_j, 3 * V_k + beta] += -1. * e_ij[beta]

    return K

def solve_arap(U, K, L, n, b, bc):
    weighted_cov = K.transpose() @ U
    vert_count = n
    R = np.zeros((3 * vert_count, 3))

    for i in range(vert_count):
        cov_i = weighted_cov[3*i:3*i+3, :]
        u, S, Vt = np.linalg.svd(cov_i)
        Ri = u @ Vt
        R[3*i:3*i+3, :] = Ri

    Aeq = sp.sparse.csc_matrix((0, 0))
    Beq = np.array([])

    B = K @ R
    ok, Z = igl.min_quad_with_fixed(L, B, b, bc, Aeq, Beq, False)

    return Z

class BIMODEL_OT_SaveModels(bpy.types.Operator):
    bl_idname = "bimodel.save_models"
    bl_label = "Save Models"
    bl_description = "Saves edit and options as obj files"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_synced()

    def execute(self, context):
        model = BiModel.last_prog
        path = os.path.join(os.getcwd(), obj_folder, model.text_manager.safe_name())
        Path(path).mkdir(parents=True, exist_ok=True)
        me = active_object()

        # Use bmesh to handle faces and edges
        bm = bmesh.new()
        bm.from_mesh(me)
        ensure_tables(bm)
        # tmp_prog = BiModel.last_prog.text_manager
        # updated_prog = tmp_prog
        
        for op in context.scene.options:
            name = op.name
            if op.dump and name in Opt.synced:
                sync, time = Opt.synced[name]
                vecs = Opt.trace_vertices(sync['x'])
                verts_from_numpy(bm, vecs)
                save_to_obj(bm, os.path.join(path, f"{name}.obj"))

                BiModel.last_prog.text_manager.update_program(sync['x'])
                prog_text = BiModel.last_prog.text_manager.program()
                txt_file = open(os.path.join(path, f"{name}.txt"), "w")
                txt_file.write(prog_text)
                txt_file.close()
                # updated_prog = tmp_prog

        BiModel.last_prog.text_manager.update_program(BiModel.last_prog.current_params)

        bm_sync = load_mesh(sync_mesh_name)
        save_to_obj(bm_sync, os.path.join(path, "Sync.obj"))

        bm.free()
        bm_sync.free()
        return {"FINISHED"}

class BIMODEL_OT_CopyDeformation(bpy.types.Operator):
    bl_idname = "bimodel.copy_deformation"
    bl_label = "Copy Deformation"
    bl_description = "Copy Deformation"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Opt.is_interpreted()

    def execute(self, context):
        bm = edit_mesh()
        bm_def = load_mesh(def_mesh_name)
        for i in range(len(bm.verts)):
            bm.verts[i].co = bm_def.verts[i].co.copy()
        update_edit_mesh()
        return {"FINISHED"}

def update_view(self, context):
    view_selected_option()

class BiModelPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_options = {"DEFAULT_CLOSED"}


class BIMODEL_PT_BiModelControls(BiModelPanel, Panel):
    bl_label = "BiModel debug controls"
    bl_idname = "BIMODEL_PT_BiModelControls"
    bl_context = "scene"

    def draw(self, context):
        pass

class BIMODEL_PT_Interpreter(BiModelPanel, Panel):
    bl_parent_id = "BIMODEL_PT_BiModelControls"
    bl_label = "Interpreter controls"

    def draw(self, context):
        layout = self.layout
        layout.activate_init = True
        row = layout.row()
        row.operator('bimodel.run', text="RUN")

        interpret = layout.row()
        interpret.alert = bpy.data.filepath == ''
        interpret.operator('bimodel.interpret', text="1. INTERPRET", icon='FILE_SCRIPT')

        tools = layout.row()

        tools.operator('bimodel.print_traces', text="Print traces")
        tools.operator('bimodel.select_edit', text="Select Edited")

        selection = layout.row()
        selection.prop(context.scene, "select_names", text="Selection", emboss=True)
        selection.operator('bimodel.copy_names', text="", icon='COPYDOWN')

class BIMODEL_PT_Stats(BiModelPanel, Panel):
    bl_parent_id = "BIMODEL_PT_Interpreter"
    bl_label = "Stats"

    def draw(self, context):
        if Opt.is_interpreted():
            layout = self.layout
            if Opt.current_params is not None:
                layout.row().label(text=f"Params: {Opt.current_params}")
            else:
                layout.row().label(text=f"Params:")

            data = edit_mesh_to_numpy()
            if data:
                V, F = data
                layout.row().label(text=f"Verts: {len(V)}")
                layout.row().label(text=f"Faces: {len(F)}")
                layout.row().label(text=f"Area: {Opt.ob.call_function('surface_area', V, F, len(F) * 3)}")
                layout.row().label(text=f"Volume: {Opt.ob.call_function('volume', V, F, len(F) * 3)}")
                center = np.zeros(3, dtype=np.float64)
                Opt.ob.call_function('center_of_mass', V, F, len(F) * 3, center)
                layout.row().label(text=f"COM: {center}")
                layout.row().label(text=f"Nodes: {Opt.trace_graph_size}")

class BIMODEL_PT_Compile(BiModelPanel, Panel):
    bl_parent_id = "BIMODEL_PT_BiModelControls"
    bl_label = "Compile"

    def draw(self, context):
        layout = self.layout
        layout.activate_init = True
        row = layout.row()
        row.operator('bimodel.compile',text="2. COMPILE", icon='FILE_CACHE')

def update_weight(self, context):
    index = bpy.context.scene.options.find(self.name)
    if Opt.is_compiled() and not context.scene.presync:
        x0 = Opt.current_params.copy()
        Opt.sync_op(self.name, x0)
    # The listener on this value updates the view
    bpy.context.scene.option_index = index

class BIMODEL_OptionProperty(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    disp_name: bpy.props.StringProperty()
    weight: bpy.props.FloatProperty(default=1.0, min=1.0, step=100., update=update_weight)
    sync: bpy.props.BoolProperty(default=True)
    dump: bpy.props.BoolProperty(default=True)

class BIMODEL_UL_Options(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        synced = Opt.synced.get(item.name, None)
        row.active = bool(synced)
        row.label(text=item.name)
        if synced:
            sync, time = synced
            row.label(text=f"loss: {round(sync['fun'], 2)}")
            row.label(text=f"secs: {round(time/1e9, 2)}")
            row.active = row.active and not math.isnan(sync['fun'])
        row.prop(item, "weight", text="Weight", emboss=False)
        row.prop(item, "sync", text="")
        row.prop(item, "dump", text="")

    def invoke(self, context, event):
        pass

class BIMODEL_UL_OptionsSimple(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        synced = Opt.synced.get(item.name, None)
        row.label(text=item.disp_name)
        row.active = bool(synced)
        row.prop(item, "sync", text="")
    
    def invoke(self, context, event):
        pass

def cancel_sync():
    model = BiModel.last_prog
    model.update_params(Opt.current_params)
    bm_sync = load_mesh(sync_mesh_name)
    bm = edit_mesh()
    for i, vert in enumerate(bm.verts):
        vert.co = bm_sync.verts[i].co.copy()
    DrawHandler.create_batch()
    bm_sync.free()

class BIMODEL_OT_History(bpy.types.Operator):
    bl_idname = "bimodel.history"
    bl_label = "History"
    bl_description = "Change current params"
    bl_options = {"REGISTER"}

    action: bpy.props.EnumProperty(
        items=(
            ('ACCEPT', "ACCEPT", ""),
            ('CANCEL', "CANCEL", "")
        )
    )

    @classmethod
    def poll(cls, context):
        return BiModel.has_model() and Opt.is_synced()

    def execute(self, context):
        if self.action == 'ACCEPT':
            model = BiModel.last_prog
            # model.update_params(model.current_params)
            Opt.update_params(model.current_params)


            Opt.render_option = False
            # Save the mesh back to obj.data
            bpy.ops.object.mode_set(mode='OBJECT')

            traced = bpy.data.objects[model.text_manager.name()]
            bm = bmesh.new()
            bm.from_mesh(traced.data)
            ensure_tables(bm)
            save_mesh(trace_mesh_name, bm)
            save_mesh(def_mesh_name, bm)
            save_mesh(sync_mesh_name, bm)
            bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
            save_mesh(tri_mesh_name, bm)
            bm.free()

            # bpy.ops.object.modifier_add(type='SUBSURF')
            # bpy.ops.object.shade_smooth()

            bpy.ops.object.mode_set(mode='EDIT')
            Opt.render_option = True
        elif self.action == 'CANCEL':
            Opt.render_option = False
            cancel_sync()
            Opt.render_option = True
        else:
            raise Exception("Unknown action")
            Opt.render_option = True


        return {"FINISHED"}


class BIMODEL_PT_Sync(BiModelPanel, Panel):
    bl_parent_id = "BIMODEL_PT_BiModelControls"
    bl_label = "Sync"

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        row = layout.row()
        row.prop(context.scene, "presync", text="Presync")
        row.operator('bimodel.handle', text="Add").action = 'ADD'
        row.operator('bimodel.handle', text="Remove").action = 'REMOVE'
        row.operator('bimodel.handle', text="Clear").action = 'CLEAR'
        
        row = layout.row()
        row.operator('bimodel.sync', text="3. SYNC ALL", icon='FILE_REFRESH').sync_all = True
        rows = len(scene.options)
        row = layout.row()
        row.template_list("BIMODEL_UL_Options", "", scene, "options", scene, "option_index", rows=rows)
        option = get_selected_option()
        if option:
            row = layout.row()
            row.operator('bimodel.sync', text="", icon='FILE_REFRESH').sync_all = False
            row.label(text=option.name)
            if get_selected_sync():
                sync, time = get_selected_sync()
                layout.row().separator()
                layout.row().label(text=f"Params: {sync['x']}")
                layout.row().prop(context.scene, "params", text="Params:")
                layout.row().label(text=f"Loss: {sync['fun']}")
                layout.row().label(text=f"Iters: {sync['iters']}")
                layout.row().label(text=f"Time (milli): {round(time / 1e6, 5)}")

        layout.row().separator()
        row = layout.row()
        cancel = row.operator('bimodel.history', text="Cancel", icon='TRASH')
        cancel.action = 'CANCEL'
        accept = row.operator('bimodel.history', text="Accept", icon='CHECKMARK')
        accept.action = 'ACCEPT'
        save = layout.row()
        save.operator('bimodel.save_models', text="Save OBJ")


class BIMODEL_PT_Deformations(BiModelPanel, Panel):
    bl_parent_id = "BIMODEL_PT_Interpreter"
    bl_label = "Deformations"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        deformations = layout.row()
        deformations.operator('bimodel.biharmonic', text="Solve Bi-harmonic")
        deformations.operator('bimodel.arap', text="Solve ARAP")
        arap_iters = layout.row()
        arap_iters.prop(obj, 'arap_iters')
        copy = layout.row()
        copy.operator('bimodel.copy_deformation', text="Copy Deformation")

class BIMODEL_PT_SimpleUI(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BiModel"
    bl_label = "BiModel Controlls"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        stages = layout.row()
        stages.operator('bimodel.interpret', text="1. Interpret")
        stages.operator('bimodel.compile', text="2. Compile")
        stages.operator('bimodel.sync', text="3. Sync").sync_all=True

        selection = layout.row()
        selection.prop(context.scene, "select_names", text="Selection", emboss=True)
        selection.operator('bimodel.copy_names', text="", icon='COPYDOWN')

        row = layout.row()
        row.prop(context.scene, "presync", text="Preview")
        row.operator('bimodel.handle', text="Add").action = 'ADD'
        row.operator('bimodel.handle', text="Remove").action = 'REMOVE'
        row.operator('bimodel.handle', text="Clear").action = 'CLEAR'

        rows = len(context.scene.options)
        layout.row().template_list("BIMODEL_UL_OptionsSimple", "", context.scene, "options", context.scene, "option_index", rows=rows)

        accept = layout.row().operator('bimodel.history', text="Accept", icon='CHECKMARK')
        accept.action = 'ACCEPT'

classes = (
    BIMODEL_OT_Run,
    BIMODEL_OT_Interpret,
    BIMODEL_OT_Compile,
    BIMODEL_OT_Handle,
    BIMODEL_OT_Sync,
    BIMODEL_OT_SelectEdit,
    BIMODEL_OT_History,
    BIMODEL_OT_SaveModels,
    BIMODEL_PT_BiModelControls,
    BIMODEL_OT_CopyNames,
    BIMODEL_OT_PrintTraces,
    BIMODEL_OT_SolveBiharmonic,
    BIMODEL_OT_SolveARAP,
    BIMODEL_OT_CopyDeformation,
    BIMODEL_PT_Interpreter,
    BIMODEL_PT_Stats,
    BIMODEL_PT_Deformations,



    BIMODEL_PT_Compile,
    BIMODEL_OptionProperty,
    BIMODEL_UL_Options,
    BIMODEL_PT_Sync,

    BIMODEL_PT_SimpleUI,
    BIMODEL_UL_OptionsSimple,
)

counter = 0

def test_draw():
    global counter
    print("test draw", counter)
    counter += 1


def register():
    from bpy.utils import register_class

    for clsname in classes:
        register_class(clsname)

    bpy.types.Object.arap_iters = bpy.props.IntProperty(default=1, min=1)
    bpy.types.Scene.options = bpy.props.CollectionProperty(type=BIMODEL_OptionProperty)
    bpy.types.Scene.option_index = bpy.props.IntProperty(update=update_view)

    bpy.types.Scene.select_names = bpy.props.StringProperty(default="")
    bpy.types.Scene.params = bpy.props.StringProperty(default="")

    bpy.types.Scene.presync = bpy.props.BoolProperty(default=False)
    bpy.app.handlers.load_post.append(handle_load)
    bpy.app.handlers.depsgraph_update_post.append(handle_update)

    # bpy.types.SpaceView3D.draw_handler_add(test_draw, (), 'WINDOW', 'POST_VIEW')
    DrawHandler.register_draw()


def unregister():
    from bpy.utils import unregister_class

    for clsname in classes:
        unregister_class(clsname)

    del bpy.types.Object.arap_iters
    del bpy.types.Scene.options
    del bpy.types.Scene.option_index

    del bpy.types.Scene.select_names
    del bpy.types.Scene.params

    bpy.app.handlers.load_post.remove(handle_load)
    bpy.app.handlers.depsgraph_update_post.remove(handle_update)

    DrawHandler.unregister_draw()

    # del bpy.types.Object.arap_iters
    # del bpy.types.Scene.options
if __name__ == "__main__":
    register()
