#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include "include/matrixTools.h"

int main(int argc, char **argv) {

	srand(time(NULL));
	
	int size, rank, i, j, k;
	
	int*  arrA,* arrB,* arrC;
	int *local_a, *local_c;
	int rows_per_proc;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	rows_per_proc = N / size;

	local_a = malloc(rows_per_proc * N * sizeof(int));
	local_c = malloc(rows_per_proc * N * sizeof(int));
	arrB = malloc(N * N * sizeof(int));

	if (rank == 0) {
		arrA = malloc(N * N * sizeof(int));
		arrC = malloc(N * N * sizeof(int));
		
		// Initialize Matrices
        	for (i = 0; i < N * N; i++) {
			arrA[i] = 2*i+N; 
			arrB[i] = 3*i+N;
        	}
	}
	// 1. Distribute rows of A to all processes
	MPI_Scatter(arrA, rows_per_proc * N, MPI_INT, local_a, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

	// 2. Broadcast the entire matrix B to all processes
	MPI_Bcast(arrB, N * N, MPI_INT, 0, MPI_COMM_WORLD);
	// 3. Local Computation
	for (i = 0; i < rows_per_proc; i++) {
		for (j = 0; j < N; j++) {
			local_c[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				local_c[i * N + j] += local_a[i * N + k] * arrB[k * N + j];
			}
		}
	}

	// 4. Gather the calculated rows back into arrC
	MPI_Gather(local_c, rows_per_proc * N, MPI_INT, arrC, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Result Matrix C (first element): %d\n", arrC[0]);
		free(arrA);
		free(arrC);
	}

	free(local_a);
	free(local_c);
	free(arrB);

	MPI_Finalize();
	return 0;
}
