# Program generator that currently generates "points" that we can work with.
# It can also do evaluation at cubic bezier parameters.

context = ["## BEGIN GENERATED BLOCK"]
par_id = 0
pt_id = 0
cb_id = 0
bz_id = 0
annotate = True

def addAnnotation(text):
    if annotate:
        context.append(f"## {text}")

def printContext():
    context.append("## END GENERATED BLOCK")
    with open("examples/cub-bez.txt","w") as f:
        f.write("\n".join(context))

class Parameter:
    def __init__(self, value):
        global context,par_id
        par_id += 1
        self.id = par_id
        self.value = value
        context.append(f"{self}={value}")

    def __str__(self):
        return f"par{self.id}"

    def __sub__(self,other):
        if not isinstance(other, Parameter):
            raise Exception("Can't sub non-parameters")
        return Parameter(f"{self}-{other}")

    def __add__(self,other):
        if not isinstance(other, Parameter):
            raise Exception("Can't add non-parameters")
        return Parameter(f"{self}+{other}")

    def __mul__(self,other):
        if not isinstance(other, Parameter):
            raise Exception("Can't mul non-parameters")
        return Parameter(f"{self}*{other}")

## This doesn't work, because it results in AST nodes that are
## 1.0 - 1.0 (or similar) when added together. The system doesn't
## like this, which is really not great, because, for example, I
## want to fix the parameter the bezier curve is being sampled at,
## and only derive w/r/t the curve's points/tangents as parameters.
class FixedParameter(Parameter):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return f"{self.value}"

class Point:
    def __init__(self, x, y, z):
        global context,pt_id
        pt_id += 1
        self.id = pt_id
        self.x =f"pt{self.id}x"
        self.y =f"pt{self.id}y"
        self.z =f"pt{self.id}z"
        context.append(f"{self.x}={x}\n{self.y}={y}\n{self.z}={z}")

    def __str__(self):
        return f"{self.x},{self.y},{self.z}"

    def __add__(self,other):
        if not isinstance(other, Point):
            raise Exception("Can't add non-point")
        return Point(
            f"{self.x}+{other.x}",
            f"{self.y}+{other.y}",
            f"{self.z}+{other.z}")

    def __sub__(self,other):
        if not isinstance(other, Point):
            raise Exception("Can't sub non-point")
        return Point(
            f"{self.x}-{other.x}",
            f"{self.y}-{other.y}",
            f"{self.z}-{other.z}")

    def __mul__(self,other):
        if not isinstance(other, Parameter):
            raise Exception("Can't mul pt by non-parameter")
        return Point(
            f"{self.x}*{other}",
            f"{self.y}*{other}",
            f"{self.z}*{other}")

    def flip(self,other):
        if not isinstance(other, Point):
            raise Exception("Can't flip non-point")
        return self + (self-other)

class Bezier:
    def __init__(self, p1 : Point, p2 : Point, p3 : Point, p4 : Point):
        global bz_id
        bz_id += 1
        self.id = bz_id
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def evaluate(self, t1 : Parameter):
        addAnnotation(f"Evaluating bezier_{self.id} at ({t1})")

        ti = Parameter(f"1.0-{t1}")
        t2i = ti*ti
        t3i = t2i*ti
        t2 = t1*t1
        t3 = t2*t1

        m1 = Parameter(f"3.0*{t2i}*{t1}")
        m2 = Parameter(f"3.0*{ti}*{t2}")
        result = (self.p1 * t3i) + (self.p2 * m1) + (self.p3 * m2) + (self.p4 * t3)
        addAnnotation(f"Finished bezier_{self.id} at ({t1})")
        return result

class Cube:
    def __init__(self, pt : Point, size: float = 0.1):
        global context,cb_id
        cb_id += 1
        self.id = cb_id
        self.pt = pt
        context.append(f"C{self.id} = Cube(size={size},location=({pt}))")


annotate = False

# This example generates about 550 lines of code in our DSL. It takes a
# long time to interpret and differentiate, but sync is kind of OK.
root = Point(0.0,0.0,0.0)
s0 = root
t0 = Point(0.0, -1.0, 0.0)    # Can constrain as function of root if needed.
s1 = Point(1.0, -1.0, 0.0)
t1 = Point(2.0, -1.0, 0.0)    #
s2 = Point(2.0, -0.5, 0.5)
t2 = Point(3.0, -0.5, 1.0)    # Keep above to make the line smooth
s3 = Point(3.0, -1.0, 1.0)
t3 = Point(3.0, -1.0, 1.0)    #
s4 = Point(4.0,  0.0, 1.0)
t4 = Point(4.0,  1.0, 1.0)
b1 = Bezier(s0,t0,s1.flip(t1),s1)
b2 = Bezier(s1,t1,s2.flip(t2),s2)
b3 = Bezier(s2,t2,s3.flip(t3),s3)
b4 = Bezier(s3,t3,s4.flip(t4),s4)
controls = [s0,t0,s1,t1,s2,t2,s3,t3,s4,t4]
curves = [b1,b2,b3,b4]

sample_density = 5  ## <-- Do not increase too far!
for b in curves:
    for i in range(1,sample_density):
        ## This is the one that should be a FixedParameter
        pt = b.evaluate(Parameter(i/float(sample_density)))
        Cube(pt)
for ctrl in controls:
   Cube(ctrl,0.2)

printContext()

