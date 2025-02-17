#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void main() {

  int nthreads, thread_num;
  
#ifdef _OPENMP
  nthreads = omp_get_num_threads();
  thread_num = omp_get_thread_num();
  printf("[thread %d] Outside of parallel region = %d\n", thread_num, nthreads);
#pragma omp parallel
{
  nthreads = omp_get_num_threads();
  thread_num = omp_get_thread_num();
  printf("[thread %d] Inside of parallel region = %d\n", thread_num, nthreads);
}
#else
  printf("OpenMP not enabled.\n");
#endif
}
