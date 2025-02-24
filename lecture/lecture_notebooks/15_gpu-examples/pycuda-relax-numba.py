from sys import argv
from pylab import *
from numba import *
from numba import cuda
import numpy as np
from timeit import default_timer as time
import os

"""
    A function to pause the program 
"""


def pause():
    wait = input("Press enter to continue")


"""
    A function to clear the terminal screen. 
"""


def clear():
    os.system("cls" if os.name == "nt" else "clear")


"""
    Instructions for CUDA:

        Go to https://store.continuum.io/cshop/anaconda/
        and download Anaconda, (it is free)

        Then through the same website go to and install 
        Anaconda Accelerate, (Not free but free academic license) 
        here: https://store.continuum.io/cshop/academicanaconda

        Follow directions (Easy install) 

        Enjoy! :)
"""


# tpb is the thread count per block being operated on by the gpu
tpb = 16

"""
    function get_max is a basic function for determining the 
    larger of two values. While this is unnecessary for python
    when writing CUDA code python functions and descriptors do
    not work, so the must be rewritten.

    We see that this is a CUDA function from the cuda.jit 
    "just in time" compiler command at the beginning of the
    definition.

    This function returns an 8-bit float and needs two 8-bit floats 
    as arguments.

    device=True indicates that this is a function to be computed 
    on the gpu device as opposed to the cpu host

    inline=True forces the function to be inline in the CUDA kernel
"""


@cuda.jit(f8(f8, f8), device=True, inline=True)
def get_max(a, b):
    if a > b:
        return a
    else:
        return b


"""
    This is the main CUDA kernel.
    CUDA operation must be initialized by calling a CUDA kernel, 
    the kernel is a function definition that operates only on the 
    CUDA device. 

    A CUDA kernel cannot return anything. 

    This is declared to the system as a CUDA kernel through the 
    jit argument given prior to the function definition. 

    You will notice that all passed in variables begin with 'd_'
    this is a naming convention in order to keep track of variables
    on the 'd'evice as compared to variables on the 'h'ost.

    CUDA kernels are to be written is a serial context. 
    When invoked the kernel launches on N threads in the gpu
    each thread uses the code serially and only when combined 
    with all other threads does it become parallel. 
"""


@cuda.jit(void(f8[:, :], f8[:, :], f8[:, :], f8[:, :]))
def c_relaxer(d_present, d_next, d_fixed, error):

    # Declare a shared array to store the error values, the changes
    err_sm = cuda.shared.array((tpb, tpb), dtype=f8)

    """
        CUDA threads inherently are able to keep track of their 
        positions in the thread block, essentially a self aware 
        array of CUDA threads. 

        CUDA threads come standard with 3-dimensional ID calls, 
        we can use one, two or all three at will. 

        These calls assign variables to the thread locations, 
        in this case we have x and y directions for our 2-dimensional 
        arrays. 
    """
    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    """
        CUDA blocks also have their own tracking calls, one 3-dim 
        vector for the block ID, and one 3-dim vector for the block 
        dimensions. 

        We establish the total size of the grid with these equations
        (for a 2 dimensional array)
        x = threadIdx.x + blockIdx.x * blockDim.x
        y = threadIdx.y + blockIdx.y * blockDim.y

        These formulas are used so frequently that a shortcut was made:
        cuda.grid('number of dimensions')
    """
    i, j = cuda.grid(2)

    # Finding the shape of the array to be calculated.
    n = d_present.shape[0]
    m = d_present.shape[1]

    err_sm[ty, tx] = 0
    # Stay within the edges
    if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
        # if the corresponding fixed value is zero then calculate the new value
        if d_fixed[j, i] == 0:
            d_next[j, i] = 0.25 * (
                d_present[j, i + 1]
                + d_present[j, i - 1]
                + d_present[j + 1, i]
                + d_present[j - 1, i]
            )
            # Calculate the change
            err_sm[ty, tx] = d_next[j, i] - d_present[j, i]
        else:
            d_next[j, i] = d_present[j, i]

    """
        CUDA threads are not required to run an any specific order. In fact
        CUDA guarantees that they will be executed out of order, this poses
        a problem when trying to use data from the same locations. 

        syncthreads acts as a barrier for the threads, the GPU will not proceed
        to the next commands until all previous commands on all threads have been
        completed. 
    """
    cuda.syncthreads()

    """
        This chunk of code comes directly from an example I found online, 
        I am not 100% sure what they are specifically for but I believe 
        that they are pulling out the largest error/change in each thread
        block and assigning it to the error grid. 
    """
    # max-reduce err_sm vertically
    t = tpb // 2
    while t > 0:
        if ty < t:
            err_sm[ty, tx] = get_max(err_sm[ty, tx], err_sm[ty + t, tx])
        t //= 2
        cuda.syncthreads()

    # max-reduce err_sm horizontally
    t = tpb // 2
    while t > 0:
        if tx < t and ty == 0:
            err_sm[ty, tx] = get_max(err_sm[ty, tx], err_sm[ty, tx + t])
        t //= 2
        cuda.syncthreads()

    if tx == 0 and ty == 0:
        error[by, bx] = err_sm[0, 0]


def main():
    # Clear the console screen
    clear()

    # Get the filename
    try:
        filename = argv[1]
        # Open the file
        fin = open(filename, "r")

        # Set up the initialArray Array
        initialArray = loadtxt(fin, unpack=True)
        # print(initialArray)
    except: 
        N = 512
        initialArray = np.random.random((N,N))
        #print("Please enter the name of the file to initialArray.")
        #filename = input(">>>")


    # print("DEBUG: ", len(initialArray))
    # print("DEBUG: ", len(initialArray[0]))
    # print("DEBUG: ", initialArray[0][0])

    # Set up the h_fixed array
    lenX = len(initialArray[0])
    lenY = len(initialArray)
    h_fixed = zeros([lenY, lenX])

    for i in range(lenY):
        for j in range(lenX):
            h_fixed[i, j] = float(initialArray[i][j])

    # Create the h_present array
    h_present = copy(initialArray)

    # and the next array
    h_next = zeros([lenY, lenX])

    # Establish the block dimensions, a 2-D array of length tpb
    blockdim = (tpb, tpb)

    # Establish the overall CUDA grid dimensions.
    # By declaring the dimensions in this manner the code becomes
    #   instantly scalable, if you move to a different computer with
    #   more CUDA cores available it will automatically scale up.
    griddim = (lenX // blockdim[0], lenY // blockdim[1])

    error_grid = zeros(griddim)

    """
        The CUDA stream is a trail of commands that get sent to the GPU.
        While not necessary the stream helps to ensure that commands 
        get processed in the correct order. 

        If multiple GPUs are available you can establish multiple streams
        and send commands to each GPU. 
    """
    stream = cuda.stream()

    """
        The GPU can only perform functions on information that is in it's 
        memory, so we have to allocate that memory for what we need. 

        We create usable device variables by sending our host variables 
        'to_device'
    """
    d_present = cuda.to_device(h_present, stream=stream)
    d_next = cuda.to_device(h_next, stream=stream)
    d_fixed = cuda.to_device(h_fixed, stream=stream)
    d_error_grid = cuda.to_device(error_grid, stream=stream)

    # stream.synchronize()

    # set up the stopping point for the calculations
    tolerance = 1e-6
    error = 1.0

    # Using a built in timer feature we keep track of how long the process
    # takes by subtracting the start time from the end time.
    start = time()

    # We also keep track of how many times the function iterates
    iter = 0

    print("Beginning GPU calculations")

    # This while loop is the iterations for the GPU calculations
    while error > tolerance:

        # print(iter)
        """
        Here we have the kernel call for the GPU function.
        As you can see the kernel requires the block and thread
        dimensions in square brackets, also we can see that it
        is a part of our stream.

        We then pass the desired variables, device variables only,
        to the kernel.
        """
        c_relaxer[griddim, blockdim, stream](d_present, d_next, d_fixed, d_error_grid)

        # Like before, in order to do any operations with the CPU we have to
        #   pass the variable back to the host.
        # This is done in order to mitigate the while loop
        d_error_grid.to_host(stream)

        stream.synchronize()

        error = abs(error_grid.max())

        # This print statement is for confirmation that the program is running
        #   we can further increase the speed by eliminating this functionality.
        # print error
        iter += 1

        # Swap the present and next array for the next set of calculations.
        tmp = d_present
        d_present = d_next
        d_next = tmp

    end = time()
    cudaTime = end - start
    print("%d iterations in %0.5f seconds" % (iter, cudaTime))

    d_present.copy_to_host(h_present, stream)

    stream.synchronize()

    # print h_present
    figure()

    CP = contourf(h_present)
    title("CUDA: %d by %d array, %0.5f seconds" % (lenX, lenY, cudaTime))

    filename = filename.split(".")
    name = filename[0]

    # show()
    savefig(name + "_%0.2f.png" % (cudaTime))


if __name__ == "__main__":
    main()
