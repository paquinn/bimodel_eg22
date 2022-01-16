import functools
import operator
import sys
import math
from time import perf_counter_ns
from abc import ABC
import numbers
import numpy as np
import igl


if __name__ == "__main__":
  import crun
  from nodes import *
else:
  from .nodes import *

class DiffGraph():
  def __init__(self, roots, losses):
    self.context = DiffContext()
    self.roots = roots
    self.losses = losses
  def clone(self):
    roots, loss = self.context.clone(self.roots, self.losses)
    return DiffGraph(roots, loss)
  
class DiffContext:
  graphID = 1

  def __init__(self):

    # Mapping from path (set of diff nodes) -> computation topology.
    # This is so we don't need to rerun DFS to determine the propagation order
    # when the graph hasn't actually changed!
    self.pathCache = {}
    # TODO: (Dan) Optimization: account for `current_uses` via some visited boolean
    # from clean. This would mean we _dont have to propagate values to parts of the
    # graph that aren't being used_. Would definitely need to clear the path cache
    # when that is done, though.

    # Keep a cache of each op's depth from the root. This can't change unless
    # we allow deleting nodes _and_ reattaching them.
    self.depthCache = {}

  # Actually execute the value propagation through the diff graph once
  # all of the root parameter values have been updated.
  @staticmethod
  def dispatch(seen, ordering, mode):
    if mode == 'DIRECT':
      for node in ordering:
        if not (node in seen):
          node.recompute()
    # For computing forward derivatives
    elif mode == 'PARTIAL':
      for node in ordering:
        if not (node in seen):
          node.partial = node.forward()

  @staticmethod
  def nameGradient(roots):
    result = { root.name : root.gradient for root in roots }
    for root in roots:
      root.gradient = 0.0
    return result

  @staticmethod
  def referenceGradient(roots):
    result = { root : root.gradient for root in roots }
    for root in roots:
      root.gradient = 0.0
    return result

  @staticmethod
  def arrayGradient(roots):
    gradient = [0.0] * len(roots)
    for (i,root) in enumerate(roots):
      gradient[i] = root.gradient
      root.gradient = 0.0
    return gradient

  # Since update is responsible for computing a forward sweep, we repurpose it for
  # computing forward partials. The path-cache means that we don't have to recompute
  # the update topology on multiple subsequent calls.
  def update(self, mapping : dict, mode = 'DIRECT', log = False):
    updatedNodes = []
    if mode == 'DIRECT':
      for node, newValue in mapping.items():
        node.primal = newValue
        updatedNodes.append(node)
    elif mode == 'PARTIAL':
      for node, newValue in mapping.items():
        node.partial = newValue
        updatedNodes.append(node)
    else:
      raise Exception(f"Unknown mode for graph update: '{mode}'")

    updatedNodes = tuple(updatedNodes)
    seen = set(updatedNodes)
    if log:
      print(f"Updating: { { n:mapping[n] for n in updatedNodes } }")
    if updatedNodes in self.pathCache:
      order = self.pathCache[updatedNodes]
      DiffContext.dispatch(seen,order,mode)
      return

    # Goal: obtain all of the downstream nodes that will need to be recomputed.
    # This does not account for memoization, i.e. we might not need to recompute
    # a node during one of these passes if none of its dependencies have actually
    # changed, as such. That said, I'm not sure it's that common a case -- the
    # reason we're batch-updating parameters anyway is because everything is changing.
    order = []
    permanent = set()
    temporary = set()

    if log:
      for origin in seen:
        print(f"{origin}, uses = {[str(use) for use in origin.uses]}")

    def visit(node):
      if node in permanent: return
      if node in temporary: raise Exception("Cyclic dependency in autodiff.")

      temporary.add(node)
      for dependent in node.uses:
        visit(dependent)
      temporary.remove(node)
      permanent.add(node)
      order.append(node)

    for origin in seen:
      visit(origin)

    order.reverse()
    if log:
      print(f"Topologized to order {[str(o) for o in order]}")
    self.pathCache[updatedNodes] = order
    DiffContext.dispatch(seen,order,mode)

  def computePartials(self, roots, traces, seed = 1.0):
    # Compute a forward pass for each of our parameters.
    tracePartials = []
    for i in range(len(roots)):
      # Initialize the seeds and propagate a context update
      self.update({ root : seed if i == j else 0.0 for (j,root) in enumerate(roots) }, mode = 'PARTIAL')
      # Collect the results of our forward propagation pass.
      if i == 0:
        tracePartials = [[[t.partial] for t in trace] for trace in traces]
      else:
        for (j,trace) in enumerate(traces):
          for (k,t) in enumerate(trace):
            tracePartials[j][k].append(t.partial)
    # Returns a matrix in which each trace coordinate (normally: [[x1,y1,z1],[x2,y2,z2]]  )
    # has instead been replaced with a list of the partials for the matrix at that coordinate,
    # so for parameters a,b: [[[x1/da, x1/db],[y1/d1,y1/db] ... and so on. ]]
    # I _think_ this is the same shape that `swapped` used to be.
    return tracePartials

  def clean(self, node):
    return self.cleanNodes([node])

  @staticmethod
  def verifyVersion(root, version):
    rooted = set([root])
    nextList = list(rooted) # A queue is better here, but OK.

    # Mark pass
    while len(nextList) > 0:
      current = nextList.pop()
      for parent in current.parents:
        if not (parent in rooted):
          rooted.add(parent)
          nextList.append(parent)
          assert parent.graphID == version


  @staticmethod
  def computeProvenanceWeights(selected, roots, traces):
    # Assumes selected is an array of selected vertex indices
    selected = set(selected)

    # Mark each Diff object as a part of vertex if it is one, and which.
    # We don't need to clear this state, because if the vertices
    # change that necessarily means the program has been reinterpreted
    # and there are fresh Diff objects.
    for (i,coords) in enumerate(traces):
      for coord in coords:
        coord.vertexIndices.add(i)

    # The set of vertex indices touched by this variable
    vertices = { var: set() for var in roots }

    # For each of the variables in the program, DFS down through
    # the computation graph, denote which vertices they access.
    # Note: this would be trivial to parallelize if it takes a long time.
    #       it's quadratic, so it well might.
    stack = []
    seen = set()
    for var in roots:
      seen.clear()
      stack.append(var)
      while len(stack) != 0:
        current = stack.pop()
        if current in seen:
          continue
        seen.add(current)
        vertices[var] |= current.vertexIndices
        for use in current.uses:
          stack.append(use)

    # print(f"Vertices = {[ (var.name, sorted(list(vertices[var]))) for var in roots ]}")
    # print(f"Selected: {sorted(list(selected))}")

    weights = [ 0.0 for var in roots ]
    for (i,var) in enumerate(roots):
      selectedVerts = 0.0
      for vtx in vertices[var]:
        if vtx in selected:
          selectedVerts += 1.0
      if len(vertices[var]) == 0:
        weight = 0.0
      else:
        weight = 1.0 - (selectedVerts / float(len(vertices[var])))
      weights[i] = weight*weight

    # Using min(*weights) breaks for single weight
    minweight = min(weights)
    diff = 1.0 - minweight
    # print(f"MinWeight: {minweight}, Weights = {[ (roots[i].name, w) for (i,w) in enumerate(weights)]}")
    if diff != 0.0:
      weights = [ (w-minweight) * (1.0/diff) for w in weights ]
    return weights

  # Clone is like an aggressive version of clean. You'd want to use it when
  # you're going to create a really expensive loss function that you don't
  # want to interfere with any later losses.
  #
  # - A: Find all of the nodes used in a given loss.
  #      Importantly, we return a tree with the same # of root parameters.
  # - B: Construct an equivalent computation graph.
  # We will keep a cache of things that have been cloned.
  def clone(self, roots, original_losses):
    DiffContext.graphID += 1
    graphID = DiffContext.graphID

    isSingleLoss = isinstance(original_losses,Diff)
    if isSingleLoss:
      losses = [original_losses]
    else:
      losses = original_losses
    rooted = set(losses) | set(roots)
    # print(f"Rooted: {[str(r) for r in rooted]}")
    mapping = dict()

    nextList = list(rooted)
    # version = nextList[0].graphID
    # for node in nextList:
      # DiffContext.verifyVersion(node, version)

    def visit(node):
      if node in mapping:
        return mapping[node]
      if all([parent in mapping for parent in node.parents]):
        parents = [mapping[parent] for parent in node.parents]
        newNode = node.clone(*parents)
        # newNode.graphID = graphID
        for parent in parents:
          assert newNode in parent.uses
        mapping[node] = newNode
      else:
        nextList.append(node)
        for parent in node.parents:
          if parent not in mapping:
            nextList.append(parent)

    while len(nextList) != 0:
      next = nextList.pop()
      visit(next)
    # for node in mapping.values():
      # DiffContext.verifyVersion(node, graphID)

    newRoots = [mapping[root] for root in roots]
    newLosses  = [mapping[node] for node in losses]
    return newRoots, newLosses[0] if isSingleLoss else newLosses


  # Cleaning is required when there are potentially uses of our parameters that
  # do not end up in our loss node. This is pretty much _always_ the case if
  # we haven't selected all of the vertices -- and possibly if we have, too, since
  # even otherwise innocuous arithmetic in a user program could cause uses to be
  # registered as a side effect.
  def cleanNodes(self, nodes):
    # print(f"Cleaning: {nodes}")
    self.pathCache = {}
    rooted = set(nodes)
    nextList = list(rooted) # A queue is better here, but OK.

    # Mark pass
    while len(nextList) > 0:
      current = nextList.pop()
      for parent in current.parents:
        if not (parent in rooted):
          rooted.add(parent)
          nextList.append(parent)

    # Sweep pass. Instead of _actually_ disconnecting the node (which would screw up
    # future loss evaluations) we will instead set the offset to what we _know_ we are
    # going to need.

    # removed = 0
    for current in rooted:
      current.current_uses = len([n for n in current.uses if n in rooted])


  # Do a DFS to find the eval order. The goal here is to generate code
  # that is in the correct order, not necessarily in a particularly efficient
  # order. We're going to hand this off to gcc with -O3, which will handle
  # proper constant propagation, reordering, and dead code elimination.
  def parent_dfs(self, nodes):
    order = []
    permanent = set()
    temporary = set()

    visitStack = []
    def visit(rootNode):
      visitStack.append(rootNode)
      while len(visitStack) != 0:
        node = visitStack.pop()
        if node in permanent:
          continue
        if node in temporary:
          temporary.remove(node)
          permanent.add(node)
          order.append(node)
        else:
          visitStack.append(node)
          temporary.add(node)
          if not isinstance(node,Var):
            for parent in reversed(node.parents):
              visitStack.append(parent)

    for node in nodes:
      visit(node)

    return order

  def serialize_forward_statements(self, roots, losses):
    order = self.parent_dfs(losses)

    # Load the values from the function arg before anything else.
    result = [ root.load_value(i) for (i,root) in enumerate(roots) ]
    for node in order:
      c = node.serialize()
      if len(c) > 0:
        result.append(c)
    return result

  def serialize_forward_partials(self, roots, losses):
    order = self.parent_dfs(losses)

    result = [ root.loop_partial(i) for (i,root) in enumerate(roots) ]
    for node in order:
      c = node.serialize()
      if len(c) > 0:
        result.append(node.serialize_partial())
    return result

  # It's somewhat important that ctx.clean(loss) be called before this is.
  # We don't do it here to avoid duplicating work if clean was called for
  # other reasons.
  def serialize_gradient_statements(self, roots, losses, step = 1.0):
    stack = []
    seen = set()
    seen_types = set()

    statements = []

    # The first time we see a node, add its gradient declaration. If it
    # requires any additional functions to be compiled in order to evaluate it,
    # we also register those.
    def register(node,value=0.0):
      if node not in seen:
        seen.add(node)
        statements.append(f"double {node}_d = {value};")
      if type(node) not in seen_types:
        seen_types.add(type(node))
      stack.append(node)

    # Follow the same algorithm that autodiff uses with .d(),
    # but do so iteratively to avoid blowing the stack.
    def visit(loss):
      register(loss,value = step) # This is our "gradient step"
      while len(stack) != 0:
        node = stack.pop()
        node.visits_this_evaluation += 1
        if node.current_uses <= node.visits_this_evaluation:
          node.visits_this_evaluation = 0
          stmt = node.serialize_gradient()
          if len(stmt) > 0:
            for parent in node.parents:
              register(parent)
            statements.append(stmt)

    # Usually there will only be one. But why not handle more?
    for loss in losses:
      visit(loss)

    # When we're done, store the results
    for (i,root) in enumerate(roots):
      if root in seen:
        statements.append(root.store_gradient(i))
      else:
        statements.append(f"outputs[{i}] = 0.0;")
    return statements

  def serialize_forward(self, roots, loss, tag=""):
    decl = f"double forward{tag}(double*);"
    header = f"\nstatic double forward{tag}(double* inputs) {{\n\t"
    stmts = self.serialize_forward_statements(roots,[loss])
    stmts.append(f"return {loss};")
    body = "\n\t".join(stmts)
    return decl, header + body + "\n }\n"

  # Assumes a trace-vertex representation, i.e. we are passing in 3-tuples
  def serialize_forward_traces(self, roots, traces):
    losses = [coord for vertex in traces for coord in vertex] # Flatten
    return self.serialize_forward_array(roots, losses)

  def serialize_forward_array(self, roots, traces, tag=""):
    decl = f"void forward_array{tag}(double*, double*);"
    header = f"\nstatic void forward_array{tag}(double* inputs, double* outputs) {{\n\t"
    stmts = self.serialize_forward_statements(roots,traces)
    stmts.extend([f"outputs[{i}] = {loss};" for (i,loss) in enumerate(traces)])
    body = "\n\t".join(stmts)
    return decl, header + body + "\n }\n"

  def serialize_forward_jacobian(self, roots, traces, tag=""):
    def tabbed(stmts):
      return map(lambda s: "\t"+s, stmts)

    decl = f"void forward_jacobian{tag}(double*, double*);"
    header = f"\nstatic void forward_jacobian{tag}(double* inputs, double* outputs) {{\n"
    stmts = self.serialize_forward_statements(roots,traces)
    stmts.append(f"for (int i = 0; i < {len(roots)}; i ++) {{")
    loop_stmts = self.serialize_forward_partials(roots,traces)
    loop_stmts.extend([f"outputs[{len(traces)} * i + {i}] = {loss}_p;" for (i,loss) in enumerate(traces)])
    stmts.extend(tabbed(loop_stmts))
    stmts.append("}")
    body = "\n".join(tabbed(stmts))
    return decl, header + body + "\n }\n"


  def serialize_reverse(self, roots, loss, step = 1.0, tag = ""):
    decl = f"double reverse_derivative{tag}(double*, double*);"
    header = "\nstatic double reverse_derivative%s(double* inputs, double* outputs) {\n\t" % tag
    stmts = self.serialize_forward_statements(roots,[loss])
    back = self.serialize_gradient_statements(roots,[loss],step)
    stmts.extend(back)
    stmts.append(f"return {loss};")
    body = "\n\t".join(stmts)
    return decl, header + body + "\n }\n"

  def printParents(self, node, depth = 0):
    for parent in node.parents:
      self.printParents(parent, depth+1)
      print("  " * depth + f"{parent}")
    if depth == 0:
      print(f"{node}")


  def depth(self,node):
    stack = [(node,0)]
    while len(stack) != 0:
      current, idx = stack.pop()
      if current in self.depthCache:
        continue
      if idx >= len(current.parents):
        if len(current.parents) == 0:
          result = 1
        else:
          # All cached
          parentDepths = [self.depthCache[parent] for parent in current.parents]
          result = max(parentDepths) + 1
        self.depthCache[current] = result
      else:
        stack.append((current,idx+1))
        stack.append((current.parents[idx],0))

    return self.depthCache[node]

def diffVar(name,value):
  return Diff.v(name,value)
def diffConst(value):
  return Diff.const(value)

def lift(value):
  if isinstance(value,Diff):
    return value
  elif isinstance(value,numbers.Number):
    return Diff.const(value)
  elif isinstance(value,tuple):
    return tuple([lift(v) for v in value])
  else:
    return value

def valueOf(value):
  if isinstance(value,float) or isinstance(value,int):
    return value
  elif isinstance(value,Diff):
    return value.primal
  elif isinstance(value,tuple):
    return tuple(map(lambda d: valueOf(d), value))
  elif isinstance(value,list):
    return list(map(lambda d: valueOf(d), value))
  else:
    raise Exception(f"Cannot extract diff value from type {type(value)}")


# Ellipse = 5x^2 + y^2 + 2^y. Its center is (0,-1)
def ellipse(args):
  x,y = args[0],args[1]
  x2 = x * x
  y2 = y * y
  result = (5*x2 + y2 + 2*y)
  return result + 1 # Center is -1, we optimize towards 0, so add 1.

class Point:
  def __init__(self,x,y,z):
    self.x = x
    self.y = y
    self.z = z

  def __mul__(self,factor):
    return Point(self.x*factor,self.y*factor,self.z*factor)

  def __add__(self,other):
    return Point(self.x+other.x,self.y+other.y,self.z+other.z)

  def dist(self,other):
    dx = other.x - self.x
    dy = other.y - self.y
    dz = other.z - self.z
    return (dx*dx + dy*dy + dz*dz).sqrt()

  def __str__(self):
    return f"Pt({self.x},{self.y},{self.z})"

class Bezier:
  def __init__(self,a,tA,tB,b):
    self.a = a
    self.b = b
    self.tA = tA
    self.tB = tB

  def evaluate(self,p):
    pi = (1.0 - p)
    p2i = pi * pi
    p3i = p2i * pi
    p2 = p * p
    p3 = p2 * p
    return (self.a * p3i) + self.tA * (3 * p2i * p) + self.tB * (3 * pi * p2) + (self.b * p3)


## This is our "loss" function. We are measuring the distance between the midpoint
 ## of this bezier curve and the target point we stipulated when calling the example.
def bezier(target):
  def result(args):
    p1,p2 = Point(0,0,0), Point(0,1,0)
    tA,tB = Point(args[0],args[1],args[2]), Point(args[3],args[4],args[5])
    midPoint = Bezier(p1,tA,tB,p2).evaluate(0.5)
    return midPoint.dist(target)
  return result

# Naive encoding of gradient descent -- just fire and go.
# I'm not proposing we use this as an optimizer -- but it's an illustration of
# how you'd use the autodiff library with one. This takes some starting parameters
# (floats) and a function from Diff[] -> Diff which evaluates the loss and its
# gradient at any given point.
def contextualOptimizer(rootParameters, lossNode):
  ctx = DiffContext()
  ctx.clean(lossNode)

  n = len(rootParameters)
  stepSize = 0.05
  stepMin = 1e-8
  error = sys.float_info.max
  iters = 1
  maxIters = 10000
  t1 = perf_counter_ns()

  def hasMemberAbove(gradient):
    return any([abs(step) > stepMin for (_,step) in gradient.items()])

  while iters < maxIters and error >= stepMin:
    lossNode.d(stepSize)
    gradient = DiffContext.referenceGradient(rootParameters)
    # Check to see if optimization has bottomed out
    if (hasMemberAbove(gradient)):
      error = lossNode.primal
      # Propagate the update using the context.
      ctx.update({ r : r.primal-v for (r,v) in gradient.items() })
      iters += 1
    else:
      break

  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"Optimized in {iters} iterations [{time} ms]. Error = {error}")
  print(f"Found {len(ctx.pathCache)} entries in the context cache.")
  return [round(p.primal,3) for p in rootParameters]

# Assumes this has been compiled with the appropriate step size baked in.
def compiledOptimizer(parameters, gradient, forward, derivative):
  stepMin = 1e-8
  error = sys.float_info.max
  iters = 1
  maxIters = 10000
  t1 = perf_counter_ns()
  n = len(gradient)

  while iters < maxIters and error >= stepMin:
    error = derivative()
    for i in range(n):
      parameters[i] -= gradient[i]
    iters += 1

  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"Optimized in {iters} iterations [{time} ms]. Error = {error}")
  return [round(p,3) for p in parameters]

def wrapRoots(fn,args):
  roots = [Diff.v(i,Diff.const(value)) for (i,value) in enumerate(args)]
  loss = fn(roots)
  return roots, loss

def testContextOptimize():
  roots, loss = wrapRoots(ellipse,[2.0,2.0])
  ctx = DiffContext()
  ctx.clean(loss)
  result = contextualOptimizer(roots, loss)
  print(f"Optimization result {result} (Expected: [0.0, -1.0])")

  target = Point(0.75,0.5,0.35)
  roots, loss = wrapRoots(bezier(target),[0.0,1.0,0.0,1.0,1.0,0.0])
  ctx.clean(loss)
  r = contextualOptimizer(roots,loss)
  print(f"tA = ({r[0]},{r[1]},{r[2]}), tB = ({r[3]},{r[4]},{r[5]})")

# FFI glue for our dynamically generated C code. If _at all_ possible
# we want to reuse the parameter arrays. In our naive optimizer where
# we are doing nothing but calling the functions, literally shuffling
# the parameters back and forth between arrays is about half the time
# of the whole optimization.
def getCompiledFns(roots):
  from volume import ffi, lib

  # Preserve these two allocations. The object stays alive
  # as long as the functions do, because it's captured by the closure.
  inputs = ffi.new("double[]",len(roots))
  outpus = ffi.new("double[]",len(roots))

  def fwd():
    return lib.forward(inputs)
  def der():
    return lib.reverse_derivative(inputs,outpus)

  return inputs, outpus, fwd, der

# This routine offers about 30x speedup (in the optimize step) compared
# to the one that computes the derivative from autodiff directly. To be
# fair, that one has a bunch of other python overhead -- which, given
# the relatively small size of this autodiff example, might be a higher
# percentage of the total time than strictly evaluating it is. That said,
# I think it's reasonable to expect a solid 20x speedup from doing this,
# if we're willing to eat the compilation time. Funnily enough, even with
# the compilation time this is still about 2x faster.
def testCompiledOptimize():
  ctx = DiffContext()
  start = [0.0,1.0,0.0,1.0,1.0,0.0]
  target = Point(0.75,0.5,0.35)

  t1 = perf_counter_ns()
  roots, loss = wrapRoots(bezier(target),start)
  ctx.clean(loss)
  h1, src_fwd = ctx.serialize_forward(roots, loss)
  h2, src_rev = ctx.serialize_reverse(roots, loss, step=0.05)
  crun.compiler("model", h1+h2, src_fwd+src_rev)
  p, g, fwd, der = getCompiledFns(roots)
  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"Staging & Compilation [{time} ms]")
  for i in range(len(start)):
    p[i] = start[i]
  r = compiledOptimizer(p,g,fwd,der)
  print(f"tA = ({r[0]},{r[1]},{r[2]}), tB = ({r[3]},{r[4]},{r[5]})")


def testDiffBuggy(args):
    x,y = args
    z = x * x
    _ = y - x # <-- look! unused variables! Bad.
    return (z * 4 + z + y * y + y * 2) + 2

def testMarkandSweep():
  ctx = DiffContext()
  roots, loss = wrapRoots(testDiffBuggy,[2.0,2.0])
  x,y = roots
  loss.d(1.0)
  print(f"Bad gradient:     {DiffContext.nameGradient(roots)} (Loss = {loss.primal})")
  ctx.clean(loss)
  loss.d(1.0)
  print(f"Better gradient:  {DiffContext.nameGradient(roots)} (Loss = {loss.primal})")
  ctx.update({x:-1.0, y:1.2})
  loss.d(1.0)
  print(f"Updated gradient: {DiffContext.nameGradient(roots)} (Loss = {loss.primal})")
  ctx.update({x:3.0, y:-3.0})
  loss.d(1.0)
  print(f"Updated gradient: {DiffContext.nameGradient(roots)} (Loss = {loss.primal})")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

_verts= [-0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5, -0.5 , -0.5,  0.5,  0.5, 0.5, -0.5, -0.5,  0.5, -0.5,  0.5, 0.5,  0.5, -0.5, 0.5,  0.5,  0.5]
def matMult():
  V = list(chunks(_verts,3))
  F = np.array([[1, 2, 0],[3, 6, 2],[7, 4, 6],[5, 0, 4],[6, 0, 2],[3, 5, 7],[1, 3, 2],[3, 7, 6],[7, 5, 4],[5, 1, 0],[6, 4, 0],[3, 1, 5]])
  Q = igl.harmonic_weights_integrated(np.array(V), F, 2)
  targets = [[(i+2.0)*i for i in vtx] for vtx in V]

  def sub(v1,v2): return [a-b for (a,b) in zip(v1,v2)]
  def result(args):
    verts = list(chunks(args,3))
    dists = [sub(targ,vert) for targ,vert in zip(targets,verts)]
    D = np.array(dists)
    Dt = np.transpose(D)  # make it horizontal
    DtQD = Dt @ Q.todense() @ D
    E = np.trace(DtQD)    # get the trace of the matrix (diagonal sum)
    return E
  return result

def formatGradient(gradient):
  return " ".join([f"{k}: {round(v,2)}" for (k,v) in gradient.items()])

def testNumpy():
  ctx = DiffContext()
  roots, loss = wrapRoots(matMult(),_verts)
  ctx.clean(loss)
  # hdr, res = ctx.serialize_reverse(roots,loss)
  loss.d(1.0)
  print(f"Depth: {ctx.depth(loss)}")
  print(f"Primal: {loss.primal}, Gradient: {formatGradient(DiffContext.nameGradient(roots))}")

# This function is somewhat useful when testing / debugging the AD output directly to C.
def emit(file,result,fresh=False):
  mode = "w+" if fresh else "a+"
  with open(f"{file}.c",mode) as f:
    if fresh:
      f.write("#include <math.h>\n")
    f.write(result)


def volume_triangle(v1,v2,v3):
  v321 = v3.x * v2.y * v1.z
  v231 = v2.x * v3.y * v1.z
  v312 = v3.x * v1.y * v2.z
  v132 = v1.x * v3.y * v2.z
  v213 = v2.x * v1.y * v3.z
  v123 = v1.x * v2.y * v3.z
  return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)

def volume_granular(args):
  bx,by,bz,dx,dy,dz = args
  p1 = Point(bx,    by,    bz   )
  p2 = Point(bx,    by,    bz+dz)
  p3 = Point(bx,    by+dy, bz   )
  p4 = Point(bx+dx, by,    bz   )
  total = 0.0
  total += volume_triangle(p1,p2,p3)
  total += volume_triangle(p4,p2,p1)
  total += volume_triangle(p3,p2,p4)
  total += volume_triangle(p3,p4,p1)
  return (total-2.0)*(total-2.0)

def volume_condensed(args):
  bx,by,bz,dx,dy,dz = args
  p1 = Point(bx,    by,    bz   )
  p2 = Point(bx,    by,    bz+dz)
  p3 = Point(bx,    by+dy, bz   )
  p4 = Point(bx+dx, by,    bz   )
  total = 0.0
  total += Volume([p1.x,p1.y,p1.z,p2.x,p2.y,p2.z,p3.x,p3.y,p3.z])
  total += Volume([p4.x,p4.y,p4.z,p2.x,p2.y,p2.z,p1.x,p1.y,p1.z])
  total += Volume([p3.x,p3.y,p3.z,p2.x,p2.y,p2.z,p4.x,p4.y,p4.z])
  total += Volume([p3.x,p3.y,p3.z,p4.x,p4.y,p4.z,p1.x,p1.y,p1.z])
  return (total-2.0)*(total-2.0)

def vol_optimize():
  ctx = DiffContext()
  start = [0.0,0.0,0.0,1.0,1.0,1.0]
  rs1, l1 = wrapRoots(volume_granular,start)
  rs2, l2 = wrapRoots(volume_condensed,start)
  ctx.clean(l1)
  l1.d(1.0)
  print(f"[Granular] V:{l1.primal} G:{ctx.arrayGradient(rs1)}")
  ctx.clean(l2)
  l2.d(1.0)
  print(f"[Condense] V:{l2.primal} G:{ctx.arrayGradient(rs2)}")
  r1 = contextualOptimizer(rs1,l1)
  print(f"B = ({r1[0]},{r1[1]},{r1[2]}), D = ({r1[3]},{r1[4]},{r1[5]})")
  r2 = contextualOptimizer(rs2,l2)
  print(f"B = ({r2[0]},{r2[1]},{r2[2]}), D = ({r2[3]},{r2[4]},{r2[5]})")

def compiled_vol_optimize():
  ctx = DiffContext()
  start = [0.0,0.0,0.0,1.0,1.0,1.0]
  t1 = perf_counter_ns()
  roots, loss = wrapRoots(volume_condensed,start)
  ctx.clean(loss)
  h1, src_fwd = ctx.serialize_forward(roots, loss)
  h2, src_rev = ctx.serialize_reverse(roots, loss, step=0.05)
  crun.compiler("volume", h1+h2, src_rev+src_fwd)
  p, g, fwd, der = getCompiledFns(roots)
  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"Staging & Compilation [{time} ms]")
  for i in range(len(start)):
    p[i] = start[i]
  r = compiledOptimizer(p,g,fwd,der)
  print(f"B = ({r[0]},{r[1]},{r[2]}), D = ({r[3]},{r[4]},{r[5]})")

# Generates the C for a volume node.
def generate_volume():
  def volume_node():
      v1x = Diff.v("v1x",Diff.zero())
      v1y = Diff.v("v1y",Diff.zero())
      v1z = Diff.v("v1z",Diff.zero())
      v2x = Diff.v("v2x",Diff.zero())
      v2y = Diff.v("v2y",Diff.zero())
      v2z = Diff.v("v2z",Diff.zero())
      v3x = Diff.v("v3x",Diff.zero())
      v3y = Diff.v("v3y",Diff.zero())
      v3z = Diff.v("v3z",Diff.zero())
      v321 = v3x * v2y * v1z
      v231 = v2x * v3y * v1z
      v312 = v3x * v1y * v2z
      v132 = v1x * v3y * v2z
      v213 = v2x * v1y * v3z
      v123 = v1x * v2y * v3z
      V = (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
      return [v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z], V

  ctx = DiffContext()
  roots, loss = volume_node()
  _, b1 = ctx.serialize_forward(roots, loss)
  _, b2 = ctx.serialize_reverse(roots, loss)
  # Emits here.
  print(b1)
  print(b2)


def testDiffGraph():
  ctx = DiffContext()
  bezier_start = [0.0,1.0,0.0,1.0,1.0,0.0]
  volume_start = [0.0,0.0,0.0,1.0,1.0,1.0]

  # Clone keeps gradient evaluations completely separate, but same API
  target = Point(0.75,0.5,0.35)
  r1, l1 = wrapRoots(bezier(target),bezier_start)
  l1.d(0.1)
  print(DiffContext.referenceGradient(r1))
  r2, l2 = ctx.clone(r1,l1)
  ctx.update({r1[i]: p for (i,p) in enumerate([i+5 for i in bezier_start])})
  l1.d(0.1)
  print(DiffContext.referenceGradient(r1))
  l2.d(0.1)
  print(DiffContext.referenceGradient(r2))

  print("\n -- \n")
  r3, l3 = wrapRoots(testDiffBuggy,[2,2])
  ctx.clean(l3)
  l3.d(0.1)
  print(DiffContext.referenceGradient(r3))
  r4, l4 = ctx.clone(r3,l3)
  ctx.clean(l4)
  l4.d(0.1)
  print(DiffContext.referenceGradient(r4))

def testForwards():
  ctx = DiffContext()
  x1,x2 = Diff.v("x1",Diff.const(2.0)), Diff.v("x2",Diff.const(1.62)) # roots
  l1,l2,l3 = x1*x2+x1.sin(), x1*x2+x2.cos(), x1.sqrt()+x2  # "traces"
  result = ctx.computePartials([x1,x2],[[l1,l2,l3],[l1,l2,l3]])
  print(f"Computed partials: {result}")

  # A function, or something
  def f(args):
    x,y = args
    return [[x*x,y*y],[x*y,-x*y],[-x*y,x*y],[x+y,y-x]]
  roots, ls = wrapRoots(f,[5.0,5.0])
  result = ctx.computePartials(roots, ls)
  print(f"Computed partials: {result}")

def pass_numpy():
  h = "float copy(float*, float*, int len);"
  b = """
  static float copy(float* input, float* output, int len) {
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
      output[i] = input[i];
      sum += input[i];
    }
    return sum;
  }
  """
  t1 = perf_counter_ns()
  crun.compiler("test2", h, b)
  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"compile: {time}")
  from test2 import ffi, lib

  t1 = perf_counter_ns()
  size = 10000000
  input = np.ones(size, dtype=np.float32)
  inputs = ffi.from_buffer("float[]", input)
  output = np.zeros(size, dtype=np.float32)
  outputs = ffi.from_buffer("float[]", output)
  # NOTE (PAQ): Can also create numpy arrays from ffi buffers
  # outputs = ffi.new("float[]", size)
  # out = np.frombuffer(ffi.buffer(outputs, size * 8), dtype=np.float32)
  sum = lib.copy(inputs, outputs, size)
  t2 = perf_counter_ns()
  time = round((t2 - t1) / 1000000.0)
  print(f"time: {time}")
  print(sum)
  print(output.sum())

def load_numpy_arrays(len_input, len_output):
  from test import ffi, lib
  np_inputs =  np.zeros(len_input, dtype=np.float64)
  np_outputs =  np.zeros(len_output, dtype=np.float64)
  np_partials =  np.zeros(len_input * len_output, dtype=np.float64)
  inputs = ffi.from_buffer("double[]", np_inputs)
  outputs = ffi.from_buffer("double[]", np_outputs)
  partials = ffi.from_buffer("double[]", np_partials)

  def forward(input):
    np.copyto(np_inputs, input)
    lib.forward_array(inputs, outputs)
    return np_outputs

  def forward_jacobian(input):
    np.copyto(np_inputs, input)
    lib.forward_jacobian(inputs, partials)
    return np_partials.reshape((len_input, -1))

  return forward, forward_jacobian

def array_return_test():
  ctx = DiffContext()

  x = Diff.v("x",Diff.const(0))
  y = Diff.v("y",Diff.const(0))
  z = Diff.v("z",Diff.const(0))
  a = x
  b = x + y
  c = 2 * x + 3 * y
  roots = [x, y]
  array = [a, b, c, a * b]

  h1, src_fwd = ctx.serialize_forward_array(roots, array)
  h2, src_fwd_prt = ctx.serialize_forward_jacobian(roots, array)
  crun.compiler("test", h1+h2, src_fwd+src_fwd_prt)

  forward, jacobian = load_numpy_arrays(len(roots), len(array))

  print(forward([1, 2]))
  print(jacobian([1, 2]))

def sum_test():
  ctx = DiffContext()

  x = Diff.v("x",Diff.const(0))
  y = Diff.v("y",Diff.const(0))
  z = Diff.v("z",Diff.const(0))
  a = x
  b = x + y
  c = 2 * x + 3 * y
  loss = Diff.sum([a, b, c, z])
  roots = [x, y, z]

  ctx.clean(loss)
  h1, src_fwd = ctx.serialize_forward(roots, loss)
  h2, src_rev = ctx.serialize_reverse(roots, loss, step=0.05)
  print(src_fwd)
  print(src_rev)
  


if __name__ == "__main__":
  # testContextOptimize()
  # testCompiledOptimize()
  # testMarkandSweep()
  # testNumpy()
  # vol_optimize()
  # compiled_vol_optimize()
  # testDiffGraph()
  # testForwards()
  sum_test()



