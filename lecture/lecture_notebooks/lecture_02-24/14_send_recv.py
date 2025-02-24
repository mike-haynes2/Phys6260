import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
data = np.zeros(5).astype('int')
idx = np.arange(5).astype('int')
data[1:4] = 3*comm.rank + idx[1:4]-1
print(f"P{comm.rank:02d}/{comm.size:02d}: Initial data = {data}")

# Send to the left
if comm.rank > 0:
    comm.Send(data[1], comm.rank-1, tag=comm.rank)

# Receive from the right
if comm.rank < comm.size-1:
    buffer = np.array(0)
    comm.Recv(buffer, comm.rank+1, tag=comm.rank+1)
    data[4] = buffer

# Send to the right
if comm.rank < comm.size-1:
    comm.Send(data[3], comm.rank+1, tag=comm.rank)

# Receive from the left
if comm.rank > 0:
    buffer = np.array(0)
    comm.Recv(buffer, comm.rank-1, tag=comm.rank-1)
    data[0] = buffer

print(f"P{comm.rank:02d}/{comm.size:02d}: After send/recv = {data}")
