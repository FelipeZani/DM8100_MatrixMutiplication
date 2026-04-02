#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "include/matrixTools.h"

int main(int argc, char *argv[])
{
    int i, j, k, rank, size;
    double *a = NULL, *b = NULL, *c = NULL;
    double *aa, *cc;
    double sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int elements_per_proc = (N * N) / size;
    int rows_per_proc = N / size;

    aa = malloc(elements_per_proc * sizeof(double));
    cc = malloc(elements_per_proc * sizeof(double));
    b = malloc(N * N * sizeof(double));

    if (rank == 0) {
	       a = malloc(N * N * sizeof(double));
	       c = malloc(N * N * sizeof(double));

	       for(i=0; i<N; i++) {
		for(int j = 0;j<N;j++ ){
		    a[i*N+j] = 2*i+N; 
		    b[i*N+j] = 3*i+N;
		}
	}
	// initLinearMatrix(a, c, b, 2, 3);
    }

    //  Scatter the rows of A
    MPI_Scatter(a, elements_per_proc, MPI_DOUBLE, aa, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast the entirety of B
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //  Local Computation
    for (i = 0; i < rows_per_proc; i++) { // For each row assigned to this proc
        for (j = 0; j < N; j++) {         // For each column of B
            sum = 0;
            for (k = 0; k < N; k++) {     // Dot product
                sum += aa[i * N + k] * b[k * N + j];
            }
            cc[i * N + j] = sum;
        }
    }

    // Gather the results
    MPI_Gather(cc, elements_per_proc, MPI_DOUBLE, c, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result C[0][0]: %f\n", c[0]);
        free(a); free(c);
    }

    free(aa); free(cc); free(b);
    MPI_Finalize();
    return 0;
}
