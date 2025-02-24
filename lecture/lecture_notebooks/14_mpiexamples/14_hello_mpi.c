#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_LENGTH 512

int main(int argc, char* argv[])
{

  int rank, size;
  char hostname[MAX_LENGTH];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  hostname[MAX_LENGTH-1] = '\0';
  gethostname(hostname, MAX_LENGTH-1);
  printf("P%04d/%04d: Hello world from %s\n", rank, size, hostname);

  MPI_Finalize();

  return EXIT_SUCCESS;

}
