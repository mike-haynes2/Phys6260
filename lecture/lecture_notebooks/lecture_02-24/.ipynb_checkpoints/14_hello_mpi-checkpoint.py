import socket
from mpi4py import MPI

comm = MPI.COMM_WORLD
hostname = socket.gethostname()
print(f"P{comm.rank:02d}/{comm.size:02d}: Hello world from {hostname}")
