import cffi

# Importantly, NOT objectives.c
with open("objectives_source.c","r") as f:
  code = f.read()

headers = [
  "double surface_area(double*, int*, int);",
  "double volume(double*, int *, int);",
  "double edit(double*, double*, int*, int);",
  "void center_of_mass(double*, int*, int, double*);"
]

ffibuilder = cffi.FFI()
ffibuilder.cdef("\n".join(headers))
ffibuilder.set_source("objectives",code)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)