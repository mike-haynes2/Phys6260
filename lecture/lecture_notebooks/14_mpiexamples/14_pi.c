#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double f(double x) {
    return 4.0 / (1.0 + x*x);
}

int main(int argc, char *argv[]) {
  
    int rank, size, n;
    double x, dx, sum, pi_int, error;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Running on %d cores\n", size);
        printf("Enter the number of intervals: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    // send the number of intervals to all processors
    MPI_Bcast(&n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    dx = 1.0/n;

    // each processor uses a simple midpoint rule to compute their
    // partial sum.  We loop over all the points in our discretization of
    // [0,1], and each processor sees only every nproc point
    sum = 0.0;
    for (int i = rank; i < n; i += size) {
        x = (double(i) + 0.5) * dx;
        sum += f(x);
        //printf("i = %d, x = %g, sum = %g\n", i, x, sum);
    }
    sum = sum * dx;
    //printf("n = %d, sum = %f\n", n, sum);

    // reduce the sum across all processors -- here only processor 0 will
    // get the total sum
    MPI_Reduce(&sum, &pi_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        error = M_PI - pi_int;
        printf("pi = %.15lf, error = %.15g\n", pi_int, error);
    }

    MPI_Finalize();

  return EXIT_SUCCESS;

}