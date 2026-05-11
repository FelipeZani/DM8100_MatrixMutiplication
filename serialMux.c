#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include "include/matrixTools.h"

void matrixMux(double * matrixA, double * matrixB, double * matrixC, int N){
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
			}
		}
	}
}

int main(int argc, char *argv[]){
	if (argc != 2) {
		return 1;
	}

	int N = atoi(argv[1]);
	double * matrixA = malloc(N * N * sizeof(double));
	double * matrixB = malloc(N * N * sizeof(double));
	double * matrixC = malloc(N * N * sizeof(double));
	srand(time(NULL));
	initMatrix(matrixA, matrixB, matrixC, N, 2, 3);
	double startT = omp_get_wtime();
	matrixMux(matrixA, matrixB, matrixC, N);
	double endT = omp_get_wtime();

	printf("Size: %dx%d | Exec Time: %f s\n", N, N, (endT-startT));
	printf("First item of C: %f\n", matrixC[0]);

	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}
