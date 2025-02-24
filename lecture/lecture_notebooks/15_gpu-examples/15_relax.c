/*
   To compile, I had to install the gcc-nvptx-offload package in Ubuntu.  I used GCC v12.1.0 with the following command line

gcc-12 -Ofast -fopenacc -foffload=-lm -fno-float-store 15_relax.c -o 15_relax -lm -lcuda -fcf-protection=check -g -fopt-info-all-omp

If I set n and m above 512, it seg faults because I think it's exceeding the stack size.  You can check the stack size with "ulimit -a".

*/



#include<stdio.h>

#include <math.h>
#include <string.h>
#include <openacc.h>
#include "timer.h"

//#define n 4096
//#define m 4096

int main(int argc, char** argv)
{
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float error     = 1.0f;

    float **A, **Anew;
    float *y0;
    A = (float**) malloc(m * sizeof(float*));
    Anew = (float**) malloc(m * sizeof(float*));
    y0 = (float*) malloc(n * sizeof(float));
    for (int i = 0; i < m; i++) {
      A[i] = (float*) malloc(n * sizeof(float));
      Anew[i] = (float*) malloc(n * sizeof(float));
    }
    
    //float A[n][m];
    //float Anew[n][m];
    //float y0[n];

    //memset(A, 0, n * m * sizeof(float));
    
    // set boundary conditions
    for (int i = 0; i < m; i++)
    {
        A[0][i]   = 0.f;
        A[n-1][i] = 0.f;
    }
    
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
        A[j][0] = y0[j];
        A[j][m-1] = y0[j]*expf(-pi);
    }
    
#if _OPENACC
    acc_init(acc_device_nvidia);
#endif
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    int iter = 0;
    
#pragma omp parallel for shared(Anew)
    for (int i = 1; i < m; i++)
    {
       Anew[0][i]   = 0.f;
       Anew[n-1][i] = 0.f;
    }
#pragma omp parallel for shared(Anew)    
    for (int j = 1; j < n; j++)
    {
        Anew[j][0]   = y0[j];
        Anew[j][m-1] = y0[j]*expf(-pi);
    }

#pragma acc data copy(A[0:n][0:m]) create(Anew[0:n][0:m])
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

#pragma omp parallel for shared(m, n, Anew, A) schedule(dynamic)

#pragma acc parallel loop reduction(max:error)
// #pragma acc parallel loop // without reduction(max:error) no computation
        for( int j = 1; j < n-1; j++)
        {
	   #pragma acc loop reduction(max:error)
	// #pragma acc loop // without reduction(max:error) no computation    
         for( int i = 1; i < m-1; i++ )
            {
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
 
       
#pragma omp parallel for shared(m, n, Anew, A) schedule(dynamic)

#pragma acc parallel loop
        for( int j = 1; j < n-1; j++)
        {
	#pragma acc loop
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];    
            }
        }


        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000.f);

    for (int i = 0; i < m; i++) {
      free(A[i]);
      free(Anew[i]);
    }
    free(A);
    free(Anew);
    free(y0);
    
}
