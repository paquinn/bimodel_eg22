# This file does something very fun (and platform-specific):
# It generates some C FFI code from this string at runtime,
# compiles it, and executes it.
#
# It makes a couple important assumptions about your environment:

# Currently:
# - You are running Blender 2.82 and python >= 3.4. (Tested on 3.7.4)
#   Other versions of blender may work, other versions of python almost certainly won't.
# - You have cffi >= 1.14 installed in your local env
#
# - IF you are running MacOS:
#    - gcc is in your path.
#    - You have all of your dependencies in a conda env named "blender"
#    - If this is not the case, edit the `__blender_conda_env` variable.
#    - You have clang installed (should be default on all macs)
#
# - IF you are running windows:
#    - You have a visual studio version installed.
#    - You will NEED to update `wincompile.txt` with your own info -- get the invocations
#      by running python build.py from an anaconda shell, and replace the two lines in wincompile
#      with the lines emitted there.
#    - Replace the versions added to the system environment variables in the windows section of the script
#      for the versions actually present on your system
#    - Add C:\Program Files (x86)\Windows Kits\10\bin\$YOUR_VERSION_HERE$\x64 and \x86 to your SYSTEM environment variable.
#
# You can test whether compilation is working without opening blender by running this file, `python crun.py`

from cffi import FFI
import ctypes
import _ctypes
import time
from time import perf_counter_ns
import os
import platform
import subprocess, shlex
import numpy as np
import sys
from pathlib import Path
import bpy

SYS = platform.system()
print(f"Detected system: {SYS}")

def compiler(sourcename, header, code):
  ffibuilder = FFI()
  ffibuilder.cdef(header)
  ffibuilder.set_source(sourcename,code)

  # TODO: (Dan)
  # The below really only works at ALL on mac. The proper step here would be to run this
  # and see what commands ffibuilder.compile(verbose=True) emits on Windows, and then
  # use the same ones after detecting if we're on Mac or Windows. I anticipate unix compilation
  # is somewhat easier (invoking MSVC from within a conda shell seems like a pain) so we'll stick
  # with this for the prototyping stage.

  filename = os.getcwd() + os.sep + sourcename
  print(filename)
  ffibuilder.emit_c_code(f"{filename}.c")


  if SYS == "Darwin":
    if os.environ.get('CONDA_ENVIRONMENT'):
      if 'CONDA_ENVIRONMENT' in os.environ:
        env = os.environ['CONDA_ENVIRONMENT']
      else:
        env = "blender2.92"
      conda = env + os.environ['CONDA_PREFIX']
      print(f"Using conda environment: {conda}")
    else:
      conda = Path(sys.executable).resolve().parents[1]
      print(f"Found conda environment: {conda}")
    # Clang is ~40% faster on our workload
    # The following set of LLVM opts are probably the ones we want. to actually do this, though,
    # we'd have to emit LLVM bytecode from clang with -emit-llvm to `model.bc`, run `opt` with the passes,
    # and then finally run the GCC optimizer. However doing that isn't much faster than what we're doing now.
    # It just might generate faster code. (It still avoids register allocation; which is the big slowdown before
    # -O0 and -O1, but it does include things like dead-code-elim and constant propagation, which do wonders
    # on the type of code we're generating.)
    # Passes:
    # -scoped-noalias -assumption-cache-tracker -domtree -deadargelim -opt-remark-emitter -instcombine -basiccg -always-inline -sroa -speculative-execution -libcalls-shrinkwrap -block-freq -reassociate -lcssa -scalar-evolution -memdep -sccp -demanded-bits -bdce -postdomtree -adce -rpo-functionattrs -globaldce -float2int -loop-vectorize -alignment-from-assumptions -strip-dead-prototypes -instsimplify -verify -ee-instrument -early-cse -barrier
    os.system(f'clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O0 -Wstrict-prototypes -I{conda}/include -arch x86_64 -I{conda}/include -arch x86_64 -I{conda}/include/python3.7m -c "{filename}.c" -o "{filename}.o"')
    os.system(f'gcc -bundle -undefined dynamic_lookup -L{conda}/lib -arch x86_64 -L{conda}/lib -arch x86_64 -arch x86_64 "{filename}.o" -o "{filename}.cpython-37m-darwin.so"')
    target = f"{sourcename}.cpython-37m-darwin.so"
  elif SYS == "Windows":
    target = f"{sourcename}.cp37-win_amd64.pyd"
    os.system(f"del .\{sourcename}.cp37-win_amd64.pyd")
    os.environ["PATH"] += r""";C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\arm;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.17763\x64;C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.6.1 Tools;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\tools;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\ide;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin"""
    os.environ["INCLUDE"] = r"""C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\atlmfc\include;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.17763\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.17763\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.17763\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.17763\winrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.17763\cppwinrt;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\Include\um"""
    os.environ["LIB"] = r"""C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\ARM;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\atlmfc\lib\ARM;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\VS\lib\ARM;C:\Program Files (x86)\Windows Kits\10\lib\10.0.17763\ucrt\arm;C:\Program Files (x86)\Windows Kits\10\lib\10.0.17763\um\arm;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\lib\um\arm;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\Lib\um\arm"""
    os.environ["LIBPATH"] = r"""C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\atlmfc\lib\ARM;C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\ARM;C:\Program Files (x86)\Windows Kits\10\References"""


    with open("wincompile.txt") as f:
      wincompile, winlink = f.readlines()
      wincompile = wincompile.replace("$MODULE",sourcename)
      winlink = winlink.replace("$MODULE",sourcename)
      print("Compiling")
      c = subprocess.run(shlex.split(wincompile))
      if c.returncode != 0:
        raise Exception("Compilation Failed")
      print("Linking")
      l = subprocess.run(shlex.split(winlink))
      if l.returncode != 0:
        raise Exception("Linking failed")

  # Spin until the files exist. os.system() is blocking, but we wait for consistency just in case.
  t = 0
  while True:
    currFiles = [file for file in os.listdir() if target in file]
    if len(currFiles) > 0: # Check for the linked target.
      break
    else:
      t += 0.01
      if t < 5.0:
        time.sleep(0.01)
      else:
        raise Exception("Compile timed out [5 seconds]")

# This is... beyond sketchy. We're rolling our own import mechanism.
# This loads the compiled .so file directly, dynamically initializes
# the fields it needs, as described in option 4 here, except that we
# are on mac, so we have direct access to `dlclose()` without having
# to load `libdl.so` directly -- it's just part of `_ctypes`.
#
# https://stackoverflow.com/questions/8295555/how-to-reload-a-python3-c-extension-module
#
# This really, _really_ doesn't support windows.
def test_run(name, a:int, b:int):
  if SYS == "Darwin":
    so = ctypes.PyDLL(f"./{name}.cpython-37m-darwin.so")
    getattr(so,f"PyInit_{name}").argtypes = []
    getattr(so,f"PyInit_{name}").restype = ctypes.py_object
    example = getattr(so,f"PyInit_{name}")()
    _, lib = example.ffi, example.lib
    result = lib.foo(a,b)
    print(f"Result: {result}")
    del example
    _ctypes.dlclose(so._handle) # pylint: disable=no-member
  elif SYS == "Windows":
    so = ctypes.PyDLL(f"{name}.cp37-win_amd64.pyd")
    getattr(so,f"PyInit_{name}").argtypes = []
    getattr(so,f"PyInit_{name}").restype = ctypes.py_object
    example = getattr(so,f"PyInit_{name}")()
    _, lib = example.ffi, example.lib
    result = lib.foo(a,b)
    print(f"Result: {result}")
    del example
    _ctypes.FreeLibrary(so._handle)

# This is a slightly more modular implementation of the above.
# It's Dynamic C, which is... pretty Dicey.
class DyC:
  def __init__(self, name):

    self.name = name
    if SYS == "Darwin":
      path = bpy.utils.user_resource('SCRIPTS', "addons")
      addon_name = "bimodel"
      so = ctypes.PyDLL(f"{path}/{addon_name}/{name}.cpython-37m-darwin.so")
    else:
      so = ctypes.PyDLL(f"{name}.cp37-win_amd64.pyd")
    getattr(so,f"PyInit_{name}").argtypes = []
    getattr(so,f"PyInit_{name}").restype = ctypes.py_object
    model = getattr(so,f"PyInit_{name}")()
    self.model = model
    self.ffi = model.ffi
    self.lib = model.lib
    self.so = so

  def map_input(self, input):
    if isinstance(input, np.ndarray):
      return self.to_ffi_array(input)
    else:
      return input

  def to_ffi_array(self, np_array):
    if np_array.dtype == np.float32:
      dtype = "float[]"
    elif np_array.dtype == np.float64:
      dtype = "double[]"
    elif np_array.dtype == np.int32:
      dtype = "int[]"
    elif np_array.dtype == np.int64:
      dtype = "long[]"
    else:
      raise Exception(f"Unexpected array type {np_array.dtype}")
    return self.ffi.from_buffer(dtype, np_array)

  def call_function(self, name, *args):
    ffi_args = map(self.map_input, args)
    func = getattr(self.lib, name)
    return func(*ffi_args)

  def load_scalar_function(self, len_input, name):
    np_inputs =  np.empty(len_input, dtype=np.float64)
    inputs = self.to_ffi_array(np_inputs)
    def call(input):
          np.copyto(np_inputs, input)
          func = getattr(self.lib, name)
          return func(inputs)
    return call
    
  def load_function(self, len_input, len_output, name):
    np_inputs =  np.empty(len_input, dtype=np.float64)
    np_outputs =  np.empty(len_output, dtype=np.float64)
    inputs = self.to_ffi_array(np_inputs)
    outputs = self.to_ffi_array(np_outputs)

    def call(input):
      np.copyto(np_inputs, input)
      func = getattr(self.lib, name)
      ret = func(inputs, outputs)
      return ret, np_outputs

    return call, outputs

  def free(self):
    del self.model
    if SYS == "Darwin":
      _ctypes.dlclose(self.so._handle)  # pylint: disable=no-member
    else:
      _ctypes.FreeLibrary(self.so._handle)
      os.remove(f"{self.name}.cp37-win_amd64.pyd")

def cleanup(sourcename):
  for file in os.listdir():
    if file.startswith(sourcename):
      os.remove(file)


def cycle():
  __test_header = "int foo(int, int);"
  __test_a = "static int foo(int x, int y){ return x; }"
  __test_b = "static int foo(int x, int y){ return y; }"

  compiler("example",__test_header,__test_a)
  test_run("example",10,5)
  compiler("example",__test_header,__test_b)
  test_run("example",42,11)
  # Expect: 10, 11


if __name__ == "__main__":
  cycle()
