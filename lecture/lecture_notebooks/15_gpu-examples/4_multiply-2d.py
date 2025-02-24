try:
    import pycuda
    import pycuda.driver as cuda
except ImportError:
    RuntimeError("PyCUDA module not found")

try:
    cuda.init()
except pycuda._driver.RuntimeError as e:
    print(f"PyCUDA Runtime error: {e}")

import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Create a 4x4 matrix of random numbers
N = 4
a = np.random.randn(N,N).astype(np.float32)

# Allocate memory on the device (GPU)
a_gpu = cuda.mem_alloc(a.nbytes)

# Transfer data to GPU
cuda.memcpy_htod(a_gpu, a)

# Define GPU (cuda) kernel
kernel = """
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
"""
mod = SourceModule(kernel)
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))  # need to specify the block size

# Fetch the data from the GPU and print it
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(f"a = {a}")
print(f"a_doubled = {a_doubled}")
