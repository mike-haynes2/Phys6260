import numpy as np
from mpi4py import MPI

def f(x):
    return 4 / (1 + x*x)

comm = MPI.COMM_WORLD
if comm.rank == 0:
    print(f"Calculating pi: Running on {comm.size} cores")
    n = ""
    while not n.isdigit():
        print("Enter the number of intervals: ")
        n = input()
    n = int(n)
else:
    n = None

n = comm.bcast(n)
dx = 1/n

# Each core uses a simple midpoint rule to compute their partial sum.  We loop over all the points in our discretization of [0,1], and each core sees only every (nproc)-th point
sum = 0.0
idx = np.arange(comm.rank, n, comm.size)
x = (idx + 0.5) * dx
sum = f(x).sum() * dx
print(x,sum)

# Reduce the sum across all cores -- here only core 0 will get the total sum because we use Reduce.  If we wanted all cores to get the sum, we'd use Allreduce.

pi_int = np.array(0.0)
comm.Reduce(sum, pi_int, op=MPI.SUM, root=0)
if comm.rank == 0:
    error = np.pi - pi_int
    print(f"pi = {pi_int:.15f}, error = {error:.15g}")