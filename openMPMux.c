#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include "include/matrixTools.h"

void matrixMux_Serial(double * matrixA, double * matrixB, double * matrixC, int N){
	for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
			}
		}
	}
}

void matrixMuxIKJ_OMP(double * matrixA, double * matrixB, double * matrixC, int N){
	#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++) {
            for(int j = 0; j < N; j++) {
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
	double * matrixC_OMP = malloc(N * N * sizeof(double));
    double * matrixC_Serial = malloc(N * N * sizeof(double));

    srand(time(NULL));

	initMatrix(matrixA, matrixB, matrixC_OMP, N, 2, 3);
    initMatrix(matrixA, matrixB, matrixC_Serial, N, 2, 3);


	double startT = omp_get_wtime();
	matrixMuxIKJ_OMP(matrixA, matrixB, matrixC_OMP, N);
	double endT = omp_get_wtime();

    matrixMux_Serial(matrixA, matrixB, matrixC_Serial, N);

    double diff = check_norm(matrixC_Serial, matrixC_OMP, N);

	printf("OpenMP (IKJ) | Size: %dx%d | Exec Time: %f s\n", N, N, (endT-startT));
    if (diff == 0.0) {
        printf("Pass (Difference = %f)\n", diff);
    } else {
        printf("Fail (Difference = %f)\n", diff);
    }

	free(matrixA); free(matrixB); free(matrixC_OMP); free(matrixC_Serial);
	return 0;
}