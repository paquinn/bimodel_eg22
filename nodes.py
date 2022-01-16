import math
import sympy as sy

uuid = 0
def fresh_temp():
  global uuid
  result = uuid
  uuid += 1
  return result

# A 'language' is an extension of an algebra.
# Anything with the following properties is a Language:
#   0, to define the algebraic zero
#   v, to make a new symbolic variable
#   const, to lift a numerical value to a language value
#
#   Each variable in this language has the following ops:
#   +, -, *, / (arithmetic)
#   sin, cos, sqrt, exp, ln (unary expressions)
class Diff:
    @staticmethod
    def zero(): return Const(0.0)

    @staticmethod
    def v(name, value): return Diff.register([value],Var(name,value))

    @staticmethod
    def const(value: float): return Const(value)

    @staticmethod
    def sum(args): return Diff.register(args.copy(), Sum(args.copy()))

    def unify(self, b):
      if isinstance(b, Diff):
        return b
      if not (isinstance(b,float) or isinstance(b,int)):
        raise Exception("Cannot operate Diff with type that is not Diff | Number")
      else:
        newB = Diff.const(b)
        return newB

    @staticmethod
    def register(dependencies, result):
      for dep in dependencies:
        dep.read(result)
        result.parent(dep)
        # if dep.graphID != 0:
        #   if result.graphID == 0:
        #     result.graphID = dep.graphID
        #   else:
        #     raise Exception("Registering invalid graph versions!")
      return result
    
    @staticmethod
    def calc_size(nodes):
      visited = set(nodes)
      next = set()
      cur = nodes.copy()
      while cur:
        for node in cur:
          for parent in node.parents:
            if parent not in visited:
              next.add(parent)
          for use in node.uses:
            if use not in visited:
              next.add(use)
        cur.clear()
        cur.extend(next)
        next.clear()
        return len(visited)

    @staticmethod
    def dot(nodes, roots):
      dot = ""
      dot += "digraph {\n"
      dot += "\trankdir=RL;\n"
      dot += "\tfontname=Consolas;\n"
      dot += "\tnode [shape=record fontname=Consolas];\n"

      node_defs = ""
      node_cons = ""

      end = set(roots)
      start = set(nodes)
      visited = set(nodes)
      next = set()
      cur = nodes.copy()

      while cur:
        for node in cur:
          color = ""
          if node in start:
            color = " fillcolor=salmon style=filled"
          elif node in end:
            color = " fillcolor=cornflowerblue style=filled"
          node_defs += f"\t{node} [label=\"{node.__class__.__name__}[{node.id}]\"{color}];\n"
          for parent in node.parents:
            node_cons += f"\t{node} -> {parent};\n"
            if parent not in visited:
              visited.add(parent)
              next.add(parent)
        cur.clear()
        cur.extend(next)
        next.clear()

      dot += node_defs
      dot += node_cons
      dot += "}"

      return dot

    @staticmethod
    def dot_file(nodes, roots, name):
      dot_graph = Diff.dot(nodes, roots)

      with open(name, "w") as f:
        f.write(dot_graph)
      

    # Store our primal value (equivalent to the `.value` in `Direct`)
    # but also a derivative function of type Float -> List[(String,Float)]
    def __init__(self, direct = None, derivative = None, forward = None):
        self.id = fresh_temp()
        if direct:
          self.primal = direct()
          self.direct = direct
        else:
          self.primal = 0.0
          self.direct = lambda: 0.0

        self.partial = 0.0  # Stored partial seed
        self.forward = forward # Forward derivative expression
        self.gradient = 0.0
        # Keep track of how many times we've read from this variable overall,
        # and how many times (during a `d()` evaluation) we have been read, so
        # that we only return our real derivative after we have accumulated all
        # of the deltas together. This makes it so that we only call our derivative
        # once, thereby serving as a join point.
        self.uses = []
        self.currently_used = True       # Is this node part of the current clean()?
        self.current_uses = 0            # 'static' visits computed by clean()
        self.parents = []
        self.visits_this_evaluation = 0  # dynamic visits tracked during d() calls
        self.derivative = derivative
        self.accumulated = 0.0

        # Extra metadata
        self.graphID = 0                 # Track which clone() we're from
        self.vertexIndices = set()

    def recompute(self):
      self.primal = self.direct()

    def read(self, dependent):
      self.uses.append(dependent)

    def parent(self, parent):
      self.parents.append(parent)

    # TODO: (Dan) convert this to an iterative, rather than recursive, formulation,
    #       using a similar pattern in `ctx` to how `update()` does it. I'm pretty
    #       sure `serialize_gradient_statements` does exactly what we need structurally.
    # Accumulate all the `delta` values into one until the last one
    # we are waiting for comes in, then return the real derivative
    # with the accumulated value.
    def d(self, delta: float):
      self.visits_this_evaluation += 1
      self.accumulated += delta
      if self.current_uses <= self.visits_this_evaluation:
        self.visits_this_evaluation = 0
        self.derivative(self.accumulated)
        self.accumulated = 0.0

    def __add__(self, b):
      b = self.unify(b)
      return Diff.register([self,b],Add(self,b))
    def __sub__(self, b):
      b = self.unify(b)
      return Diff.register([self,b],Sub(self,b))
    def __mul__(self, b):
      b = self.unify(b)
      return Diff.register([self,b],Mul(self,b))
    def __div__(self, b):
      b = self.unify(b)
      return Diff.register([self,b],Div(self,b))
    def pow(self,exp):
      exp = self.unify(exp)
      return Diff.register([self,exp],Pow(self,exp))
    def __pow__(self,exp):
      return self.pow(exp)

    def __truediv__(self, b): return self.__div__(b)
    def __neg__(self): return Diff.zero() - self

    def sin(self):  return Diff.register([self],Sin(self))
    def cos(self):  return Diff.register([self],Cos(self))
    def sqrt(self): return Diff.register([self],Sqrt(self))
    def abs(self):  return Diff.register([self],Abs(self))
    def exp(self):  return Diff.register([self],Exp(self))
    def ln(self):   return Diff.register([self],Ln(self))

    def __radd__(self,b): return Diff.const(b) + self
    def __rsub__(self,b): return Diff.const(b) - self
    def __rmul__(self,b): return Diff.const(b) * self
    def __rdiv__(self,b): return Diff.const(b) / self
    def __rtruediv__(self,b): return Diff.const(b) / self

    def __str__(self):   return f"t{self.id}"
    def __repr__(self):  return f"{self.__class__.__name__}[{self.id}]({self.primal})"
    def serialize(self): return ""
    def serialize_partial(self):  return ""
    def serialize_gradient(self): return ""

    @staticmethod
    def support_functions():
      return """
double volume_fwd(double,double,double,double,double,double,double,double,double);
void volume_rev(double, double*,double*,double*,double*,double*,double*,double*,double*,double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
""","""
static double volume_fwd(double v_v1x,double v_v1y,double v_v1z,double v_v2x,double v_v2y,double v_v2z,double v_v3x,double v_v3y,double v_v3z) {
        double t37 = 0.16666666666666666;
        double t30 = 0.0;
        double t18 = v_v3x * v_v2y;
        double t19 = t18 * v_v1z;
        double t31 = t30 - t19;
        double t20 = v_v2x * v_v3y;
        double t21 = t20 * v_v1z;
        double t32 = t31 + t21;
        double t22 = v_v3x * v_v1y;
        double t23 = t22 * v_v2z;
        double t33 = t32 + t23;
        double t24 = v_v1x * v_v3y;
        double t25 = t24 * v_v2z;
        double t34 = t33 - t25;
        double t26 = v_v2x * v_v1y;
        double t27 = t26 * v_v3z;
        double t35 = t34 - t27;
        double t28 = v_v1x * v_v2y;
        double t29 = t28 * v_v3z;
        double t36 = t35 + t29;
        double t38 = t37 * t36;
        return t38;
    }

static void volume_rev(double self_d, double *input0,double *input1,double *input2,double *input3,double *input4,double *input5,double *input6,double *input7,double *input8, double *output0, double *output1, double *output2, double *output3, double *output4, double *output5, double *output6, double *output7, double *output8) {
        double v_v1x = *input0;
        double v_v1y = *input1;
        double v_v1z = *input2;
        double v_v2x = *input3;
        double v_v2y = *input4;
        double v_v2z = *input5;
        double v_v3x = *input6;
        double v_v3y = *input7;
        double v_v3z = *input8;
        double t37 = 0.16666666666666666;
        double t30 = 0.0;
        double t18 = v_v3x * v_v2y;
        double t19 = t18 * v_v1z;
        double t31 = t30 - t19;
        double t20 = v_v2x * v_v3y;
        double t21 = t20 * v_v1z;
        double t32 = t31 + t21;
        double t22 = v_v3x * v_v1y;
        double t23 = t22 * v_v2z;
        double t33 = t32 + t23;
        double t24 = v_v1x * v_v3y;
        double t25 = t24 * v_v2z;
        double t34 = t33 - t25;
        double t26 = v_v2x * v_v1y;
        double t27 = t26 * v_v3z;
        double t35 = t34 - t27;
        double t28 = v_v1x * v_v2y;
        double t29 = t28 * v_v3z;
        double t36 = t35 + t29;
        double t38 = t37 * t36;
        double t38_d = self_d;
        double t37_d = 0.0;
        double t36_d = 0.0;
        t37_d += t38_d * t36; t36_d += t37 * t38_d;
        double t35_d = 0.0;
        double t29_d = 0.0;
        t35_d += t36_d; t29_d += t36_d;
        double t28_d = 0.0;
        double v_v3z_d = 0.0;
        t28_d += t29_d * v_v3z; v_v3z_d += t28 * t29_d;
        double v_v1x_d = 0.0;
        double v_v2y_d = 0.0;
        v_v1x_d += t28_d * v_v2y; v_v2y_d += v_v1x * t28_d;
        double t34_d = 0.0;
        double t27_d = 0.0;
        t34_d += t35_d; t27_d -= t35_d;
        double t26_d = 0.0;
        t26_d += t27_d * v_v3z; v_v3z_d += t26 * t27_d;
        double v_v2x_d = 0.0;
        double v_v1y_d = 0.0;
        v_v2x_d += t26_d * v_v1y; v_v1y_d += v_v2x * t26_d;
        double t33_d = 0.0;
        double t25_d = 0.0;
        t33_d += t34_d; t25_d -= t34_d;
        double t24_d = 0.0;
        double v_v2z_d = 0.0;
        t24_d += t25_d * v_v2z; v_v2z_d += t24 * t25_d;
        double v_v3y_d = 0.0;
        v_v1x_d += t24_d * v_v3y; v_v3y_d += v_v1x * t24_d;
        double t32_d = 0.0;
        double t23_d = 0.0;
        t32_d += t33_d; t23_d += t33_d;
        double t22_d = 0.0;
        t22_d += t23_d * v_v2z; v_v2z_d += t22 * t23_d;
        double v_v3x_d = 0.0;
        v_v3x_d += t22_d * v_v1y; v_v1y_d += v_v3x * t22_d;
        double t31_d = 0.0;
        double t21_d = 0.0;
        t31_d += t32_d; t21_d += t32_d;
        double t20_d = 0.0;
        double v_v1z_d = 0.0;
        t20_d += t21_d * v_v1z; v_v1z_d += t20 * t21_d;
        v_v2x_d += t20_d * v_v3y; v_v3y_d += v_v2x * t20_d;
        double t30_d = 0.0;
        double t19_d = 0.0;
        t30_d += t31_d; t19_d -= t31_d;
        double t18_d = 0.0;
        t18_d += t19_d * v_v1z; v_v1z_d += t18 * t19_d;
        v_v3x_d += t18_d * v_v2y; v_v2y_d += v_v3x * t18_d;
        *output0 += v_v1x_d;
        *output1 += v_v1y_d;
        *output2 += v_v1z_d;
        *output3 += v_v2x_d;
        *output4 += v_v2y_d;
        *output5 += v_v2z_d;
        *output6 += v_v3x_d;
        *output7 += v_v3y_d;
        *output8 += v_v3z_d;
 }
 """

# We implement specific classes for each node type so it's easier to customise
# the behaviour of the serializations a little, and to hypothetically later be
# able to case on node type (with `isinstance()`)

class Const(Diff):
  def __init__(self, value):
    value = float(value)
    super().__init__(lambda: value, lambda delta: None, lambda: 0.0)
  def serialize(self):
    return f"double {self} = {self.primal};"
  def serialize_partial(self):
    return f"double {self}_p = 0.0;"
  def clone(self):
    return Const(self.primal)
  def sy(self):
    return self.primal

class Var(Diff):
  def __init__(self, name, value):
    self.name = name
    def d(delta):
      self.gradient = delta
    super().__init__(lambda: value.primal, d, lambda: self.partial)
  def __str__(self): return f"v_{self.name}"

  # Var doesn't have serializers, we rely on the default impl to return
  # empty, and call these explicitly, because we need to know what index
  # to use in order to load / store.
  def load_value(self, index):
    return f"double {self} = inputs[{index}];"
  def load_partial(self, index):
    return f"double {self}_p = partials[{index}];"
  def loop_partial(self, index):
    return f"double {self}_p = i == {index};"
  def store_gradient(self, index):
    return f"outputs[{index}] = {self}_d;"
  def clone(self, value):
    return Diff.register([value],Var(self.name,value))
  def sy(self):
    return sy.Symbol(self.name)


# Guide to translating our dual lambdas into real code:
# {self}       = our primal value
# {a},{b}      = a.primal, b.primal
# {self}_d     = the delta value (we have already accumulated it by now.)
# {a}_d, {b}_d = a.d(), b.d(). We "call" by directly accumulating.
# {a}_p, {b}_p = a.partial, b.partial
class Add(Diff):
  def __init__(self, a, b):
    # d/dx f(x) + g(x) =  f'(x) + g'(x)
    def forward():
      return a.primal + b.primal
    def reverse(delta):
      a.d(delta)
      b.d(delta)
    def partial():
      return a.partial + b.partial
    super().__init__(forward, reverse, partial)
  def serialize(self):
    a,b = self.parents
    return f"double {self} = {a} + {b};"
  def serialize_gradient(self):
    a,b = self.parents
    return f"{a}_d += {self}_d; {b}_d += {self}_d;"
  def serialize_partial(self):
    a,b = self.parents
    return f"double {self}_p = {a}_p + {b}_p;"
  def sy(self):
    a,b = self.parents
    return a.sy() + b.sy()
  def clone(self, a, b):
    return Diff.register([a,b],Add(a,b))

class Sum(Diff):
  def __init__(self, args):
    def forward():
      return sum(map(lambda p: p.primal, args))
    def reverse(delta):
      for p in args:
        p.d(delta)
    def partial():
      return sum(map(lambda p: p.partial, args))
    super().__init__(forward, reverse, partial)
  def serialize(self):
    return f"double {self} = {' + '.join((map(str, self.parents)))};"
  def serialize_gradient(self):
    return " ".join(map(lambda p: f"{p}_d += {self}_d;", self.parents))
  def serialize_partials(self):
    return f"double {self}_p = {' + '.join((map(lambda p: f'{p}_p', self.parents)))};"
  def sy(self):
    return sum(map(lambda p: p.sy(), self.parents))
  def clone(self, parents):
    return Diff.register(self.parents.copy(), Sum(self.parents.copy()))


# The negation of add, but with the exact same structure.
class Sub(Diff):
  def __init__(self, a, b):
    def forward():
      return a.primal - b.primal
    def reverse(delta):
      a.d(delta)
      b.d(-delta)
    def partial():
      return a.partial - b.partial
    super().__init__(forward, reverse, partial)
  def serialize(self):
    a,b = self.parents
    return f"double {self} = {a} - {b};"
  def serialize_gradient(self):
    a,b = self.parents
    return f"{a}_d += {self}_d; {b}_d -= {self}_d;"
  def serialize_partial(self):
    a,b = self.parents
    return f"double {self}_p = {a}_p - {b}_p;"
  def sy(self):
    a,b = self.parents
    return a.sy() - b.sy()
  def clone(self, a, b):
    return Diff.register([a,b],Sub(a,b))

class Mul(Diff):
  def __init__(self, a, b):
    def forward():
      return a.primal * b.primal
    # d/d(x) f(x)g(x) = f'(x)g(x) + f(x)g'(x)
    # a.primal and b.primal are f(x) and g(x) respectively, so we
    # multiply each by the scaled value and make the recursive call to d().
    def reverse(delta):
      a.d(delta * b.primal)
      b.d(a.primal * delta)
    def partial():
      return a.partial * b.primal + a.primal * b.partial
    super().__init__(forward,reverse,partial)
  def serialize(self):
    a,b = self.parents
    return f"double {self} = {a} * {b};"
  def serialize_gradient(self):
    a,b = self.parents
    return f"{a}_d += {self}_d * {b}; {b}_d += {a} * {self}_d;"
  def serialize_partial(self):
    a,b = self.parents
    return f"double {self}_p = {a}_p * {b} + {a} * {b}_p;"
  def sy(self):
    a,b = self.parents
    return a.sy() * b.sy()
  def clone(self, a, b):
    return Diff.register([a,b],Mul(a,b))

class Div(Diff):
  def __init__(self, a, b):
    def forward():
      return a.primal / b.primal
    # d/d(x) f(x)/g(x)  =  (g(x)f'(x) - f(x)g'(x)) / g(x)^2
    def reverse(delta): # Quotient Rule
      gx2 = b.primal * b.primal
      a.d(b.primal * delta / gx2)
      b.d(-(a.primal * delta / gx2))
    def partial():
      gx2 = b.primal * b.primal
      return (b.primal*a.partial - a.primal*b.partial) / gx2
    super().__init__(forward, reverse, partial)
  def serialize(self):
    a,b = self.parents
    return f"double {self} = {a} / {b};"
  def serialize_gradient(self):
    a,b = self.parents
    t = f"t{fresh_temp()}"
    return f"double {t} = {b}*{b}; {a}_d += {b} * {self}_d / {t}; {b}_d -= {a} * {self}_d / {t};"
  def serialize_partial(self):
    a,b = self.parents
    return f"double {self}_p = ({b}*{a}_p - {a}*{b}_p) / ({b}*{b});"
  def sy(self):
    a,b = self.parents
    return a.sy() / b.sy()
  def clone(self, a, b):
    return Diff.register([a,b],Div(a,b))

class Sqrt(Diff):
  def __init__(self, a):
    super().__init__(
      lambda: math.sqrt(a.primal),
      lambda delta: a.d((0.5 * delta) / math.sqrt(a.primal)),
      lambda: (a.partial * 0.5) / math.sqrt(a.primal))
  def serialize(self):
    a = self.parents[0]
    return f"double {self} = sqrt({a});"
  def serialize_gradient(self):
    a = self.parents[0]
    return f"{a}_d += (0.5 * {self}_d) / {self};"
  def serialize_partial(self):
    a = self.parents[0]
    return f"double {self}_p = ({a}_p * 0.5) / sqrt({a});"
  def sy(self):
    a = self.parents[0]
    return sy.sqrt(a.sy())
  def clone(self, a):
    return Diff.register([a],Sqrt(a))

class Abs(Diff):
  def __init__(self,a):
    super().__init__(
      lambda: math.fabs(a.primal),
      lambda delta: a.d(delta * (1.0 if (a.primal > 0.0) else (-1.0 if a.primal < 0.0 else 0.0))),
      lambda: a.partial * (1.0 if (a.primal > 0.0) else (-1.0 if a.primal < 0.0 else 0.0))
    )
  def serialize(self):
    raise Exception("serialization not implemented for abs")
  def serialize_gradient(self):
    raise Exception("serialization not implemented for abs")
  def serialize_partial(self):
    raise Exception("serialization not implemented for abs")
  def sy(self):
    a = self.parents[0]
    return sy.Abs(a.sy())
  def clone(self, a):
    return Diff.register([a],Abs(a))


class Sin(Diff):
  def __init__(self,a):
    super().__init__(
      lambda: math.sin(a.primal),
      lambda delta: a.d(delta * math.cos(a.primal)),
      lambda: math.cos(a.primal) * a.partial)
  def serialize(self):
    a = self.parents[0]
    return f"double {self} = sin({a});"
  def serialize_gradient(self):
    a = self.parents[0]
    return f"{a}_d += {self}_d * cos({a});"
  def serialize_partial(self):
    a = self.parents[0]
    return f"double {self}_p = cos({a}) * {a}_p;"
  def sy(self):
    a = self.parents[0]
    return sy.sin(a.sy())
  def clone(self, a):
    return Diff.register([a],Sin(a))

class Cos(Diff):
  def __init__(self,a):
    super().__init__(
      lambda: math.cos(a.primal),
      lambda delta: a.d(-delta * math.sin(a.primal)),
      lambda: math.sin(a.primal) * -a.partial)
  def serialize(self):
    a = self.parents[0]
    return f"double {self} = cos({a});"
  def serialize_gradient(self):
    a = self.parents[0]
    return f"{a}_d -= {self}_d * sin({a});"
  def serialize_partial(self):
    a = self.parents[0]
    return f"double {self}_p = sin({a}) * -{a}_p;"
  def sy(self):
    a = self.parents[0]
    return sy.sqrt(a.sy())
  def clone(self, a):
    return Diff.register([a],Cos(a))

class Exp(Diff):
  def __init__(self,a):
    super().__init__(
      lambda: math.exp(a.primal),
      lambda delta: a.d(delta * math.exp(a.primal)),
      lambda: a.partial * math.exp(a.primal))
  def serialize(self):
    a = self.parents[0]
    return f"double {self} = exp({a});"
  def serialize_gradient(self):
    a = self.parents[0]
    return f"{a}_d += {self}_d * {self};"
  def serialize_partial(self):
    a = self.parents[0]
    return f"double {self}_p = {a}_p * exp({a});"
  def sy(self):
    a = self.parents[0]
    return sy.exp(a.sy())
  def clone(self, a):
    return Diff.register([a],Exp(a))

class Ln(Diff):
  def __init__(self,a):
    super().__init__(
      lambda: math.log(a.primal),
      lambda delta: a.d(delta / a.primal),
      lambda: a.partial / a.primal)
  def serialize(self):
    a = self.parents[0]
    return f"double {self} = log({a});"
  def serialize_gradient(self):
    a = self.parents[0]
    return f"{a}_d += {self}_d / {a};"
  def serialize_partial(self):
    a = self.parents[0]
    return f"double {self}_p = {a}_p / {a};"
  def sy(self):
    a = self.parents[0]
    return sy.ln(a.sy())
  def clone(self, a):
    return Diff.register([a],Ln(a))

class Pow(Diff):
  def __init__(self, a, exponent):
    super().__init__(
      lambda: math.pow(a.primal, exponent.primal),
      lambda delta: a.d(delta * exponent.primal * math.pow(a.primal, exponent.primal-1)),
      lambda: a.partial * exponent.primal * math.pow(a.primal, exponent.primal-1))
  def serialize(self):
    a,exp = self.parents
    return f"double {self} = pow({a},{exp});"
  def serialize_gradient(self):
    a,exp = self.parents
    return f"{a}_d += {self}_d * {exp} * pow({a},{exp}-1.0);"
  def serialize_partial(self):
    a,exp = self.parents
    return f"double {self}_p = {a}_p * {exp} * pow({a},{exp}-1.0);"
  def sy(self):
    a,b = self.parents
    return sy.sqrt(a.sy(), b.sy())
  def clone(self,a,exp):
    return Diff.register([a,exp],Pow(a,exp))


## The volume contributions of a triangle.
class Volume(Diff):
  def __init__(self,args):
    # TODO: (Dan)
    # Is there ever a case where we need the partial for volumes?
    # We can definitely add it, we'll just need to generate the code from the staged one.
    v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z = args
    def f():
      v321 = v3x.primal * v2y.primal * v1z.primal
      v231 = v2x.primal * v3y.primal * v1z.primal
      v312 = v3x.primal * v1y.primal * v2z.primal
      v132 = v1x.primal * v3y.primal * v2z.primal
      v213 = v2x.primal * v1y.primal * v3z.primal
      v123 = v1x.primal * v2y.primal * v3z.primal
      return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
    def d(delta):
      v_v1x = v1x.primal
      v_v1y = v1y.primal
      v_v1z = v1z.primal
      v_v2x = v2x.primal
      v_v2y = v2y.primal
      v_v2z = v2z.primal
      v_v3x = v3x.primal
      v_v3y = v3y.primal
      v_v3z = v3z.primal
      t37 = 0.16666666666666666
      t30 = 0.0
      t18 = v_v3x * v_v2y
      t19 = t18 * v_v1z
      t31 = t30 - t19
      t20 = v_v2x * v_v3y
      t21 = t20 * v_v1z
      t32 = t31 + t21
      t22 = v_v3x * v_v1y
      t23 = t22 * v_v2z
      t33 = t32 + t23
      t24 = v_v1x * v_v3y
      t25 = t24 * v_v2z
      t34 = t33 - t25
      t26 = v_v2x * v_v1y
      t27 = t26 * v_v3z
      t35 = t34 - t27
      t28 = v_v1x * v_v2y
      t29 = t28 * v_v3z
      t36 = t35 + t29
      t38 = t37 * t36
      t38_d = delta
      t37_d = 0.0
      t36_d = 0.0
      t37_d += t38_d * t36
      t36_d += t37 * t38_d
      t35_d = 0.0
      t29_d = 0.0
      t35_d += t36_d
      t29_d += t36_d
      t28_d = 0.0
      v_v3z_d = 0.0
      t28_d += t29_d * v_v3z
      v_v3z_d += t28 * t29_d
      v_v1x_d = 0.0
      v_v2y_d = 0.0
      v_v1x_d += t28_d * v_v2y
      v_v2y_d += v_v1x * t28_d
      t34_d = 0.0
      t27_d = 0.0
      t34_d += t35_d
      t27_d -= t35_d
      t26_d = 0.0
      t26_d += t27_d * v_v3z
      v_v3z_d += t26 * t27_d
      v_v2x_d = 0.0
      v_v1y_d = 0.0
      v_v2x_d += t26_d * v_v1y
      v_v1y_d += v_v2x * t26_d
      t33_d = 0.0
      t25_d = 0.0
      t33_d += t34_d
      t25_d -= t34_d
      t24_d = 0.0
      v_v2z_d = 0.0
      t24_d += t25_d * v_v2z
      v_v2z_d += t24 * t25_d
      v_v3y_d = 0.0
      v_v1x_d += t24_d * v_v3y
      v_v3y_d += v_v1x * t24_d
      t32_d = 0.0
      t23_d = 0.0
      t32_d += t33_d
      t23_d += t33_d
      t22_d = 0.0
      t22_d += t23_d * v_v2z
      v_v2z_d += t22 * t23_d
      v_v3x_d = 0.0
      v_v3x_d += t22_d * v_v1y
      v_v1y_d += v_v3x * t22_d
      t31_d = 0.0
      t21_d = 0.0
      t31_d += t32_d
      t21_d += t32_d
      t20_d = 0.0
      v_v1z_d = 0.0
      t20_d += t21_d * v_v1z
      v_v1z_d += t20 * t21_d
      v_v2x_d += t20_d * v_v3y
      v_v3y_d += v_v2x * t20_d
      t30_d = 0.0
      t19_d = 0.0
      t30_d += t31_d
      t19_d -= t31_d
      t18_d = 0.0
      t18_d += t19_d * v_v1z
      v_v1z_d += t18 * t19_d
      v_v3x_d += t18_d * v_v2y
      v_v2y_d += v_v3x * t18_d
      v1x.d(v_v1x_d)
      v1y.d(v_v1y_d)
      v1z.d(v_v1z_d)
      v2x.d(v_v2x_d)
      v2y.d(v_v2y_d)
      v2z.d(v_v2z_d)
      v3x.d(v_v3x_d)
      v3y.d(v_v3y_d)
      v3z.d(v_v3z_d)
    super().__init__(f,d)
    Diff.register(args,self)
  def serialize(self):
    args = ",".join([f"{arg}" for arg in self.parents])
    return f"double {self} = volume_fwd({args});"
  def serialize_gradient(self):
    args = ",".join([f"&{arg}" for arg in self.parents] + [f"&{arg}_d" for arg in self.parents])
    return f"volume_rev({self}_d,{args});"
  def sy(self):
    return sy.Function('Volume')(*[p.sy() for p in self.parents])