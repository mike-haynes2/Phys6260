#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{

  int rank, size, sendto, recvfrom;
  int data[5], buffer;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  data[0] = data[4] = 0;
  for (int i = 1; i < 5; i++)
    data[i] = rank*3 + (i-1);

  // Send to the left and receive from the right
  //
  // Send: rank -> rank-1
  // Receive: rank <- rank+1

  if (rank == 0)
    sendto = MPI_PROC_NULL;
  else
    sendto = rank-1;

  if (rank == size-1)
    recvfrom = MPI_PROC_NULL;
  else
    recvfrom = rank+1;

  MPI_Sendrecv(data+1, 1, MPI_INTEGER, sendto, 1,
    data+4, 1, MPI_INTEGER, recvfrom, 1,
    MPI_COMM_WORLD, &status);

  // Send to the right and receive from the left
  //
  // Send: rank -> rank+1
  // Receive: rank <- rank-1

  if (rank == size-1)
    sendto = MPI_PROC_NULL;
  else
    sendto = rank+1;

  if (rank == 0)
    recvfrom = MPI_PROC_NULL;
  else
    recvfrom = rank-1;

  MPI_Sendrecv(data+3, 1, MPI_INTEGER, sendto, 1,
    data, 1, MPI_INTEGER, recvfrom, 1,
    MPI_COMM_WORLD, &status);

  printf("[rank = %d] data = %d %d %d %d %d\n", rank, data[0], data[1], data[2], data[3], data[4]);

  MPI_Finalize();

  return EXIT_SUCCESS;

}
