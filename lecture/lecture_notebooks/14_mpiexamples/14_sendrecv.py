import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
data = np.zeros(5).astype('int')
idx = np.arange(5).astype('int')
data[1:4] = 3*comm.rank + idx[1:4]-1
print(f"P{comm.rank:02d}/{comm.size:02d}: Initial data = {data}")

# Send to the left and receive from the right
#
# Send: rank -> rank-1
# Receive: rank <- rank+1
    
sendto = MPI.PROC_NULL if comm.rank == 0 else comm.rank-1
recvfrom = MPI.PROC_NULL if comm.rank == comm.size-1 else comm.rank+1
buffer = np.array(0)

comm.Sendrecv(data[1], sendto, recvbuf=buffer, source=recvfrom)
data[4] = buffer

print(f"P{comm.rank:02d}/{comm.size:02d}: After left-send = {data}")

# Send to the right and receive from the left
#
# Send: rank -> rank+1
# Receive: rank <- rank-1
    
sendto = MPI.PROC_NULL if comm.rank == comm.size-1 else comm.rank+1
recvfrom = MPI.PROC_NULL if comm.rank == 0 else comm.rank-1
buffer = np.array(0)

comm.Sendrecv(data[3], sendto, recvbuf=buffer, source=recvfrom)
data[0] = buffer

print(f"P{comm.rank:02d}/{comm.size:02d}: After right-send = {data}")
