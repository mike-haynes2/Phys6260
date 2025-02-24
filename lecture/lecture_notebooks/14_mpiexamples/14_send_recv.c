#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{

  int rank, size;
  int data[5], buffer;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  data[0] = data[4] = 0;
  for (int i = 1; i < 5; i++)
    data[i] = rank*3 + (i-1);

  // send to the left
  if (rank > 0)
    MPI_Send(data+1, 1, MPI_INTEGER, rank-1, rank, MPI_COMM_WORLD);

  // receive from the right
  if (rank < size-1)
    MPI_Recv(data+4, 1, MPI_INTEGER, rank+1, rank+1, MPI_COMM_WORLD, &status);

  // send to the right
  if (rank < size-1)
    MPI_Send(data+3, 1, MPI_INTEGER, rank+1, rank, MPI_COMM_WORLD);
  
  // receive from the left
  if (rank > 0)
    MPI_Recv(data, 1, MPI_INTEGER, rank-1, rank-1, MPI_COMM_WORLD, &status);

  printf("[rank = %d] data = %d %d %d %d %d\n", rank, data[0], data[1], data[2], data[3], data[4]);

  MPI_Finalize();

  return EXIT_SUCCESS;

}
