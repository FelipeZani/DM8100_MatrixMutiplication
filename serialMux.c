#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include "include/matrixTools.h"
#define N 1000

void matrixMux(int ** matrixA, int ** matrixB, int ** matrixC){
	for(int i = 0; i < N; i++) {
        	for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
}

int main(){
	
	int ** matrixA = malloc(sizeof(int*)*N);
	int ** matrixB = malloc(sizeof(int*)*N);
	int ** matrixC = malloc(sizeof(int*)*N);
	
     	srand(time(NULL));
	

	initMatrix(matrixA, matrixB,matrixC,2,3);
	double startT = omp_get_wtime();
	double endT;

	matrixMux(matrixA,matrixB,matrixC);
	endT = omp_get_wtime();

	printf("Exec Time: %f", (endT-startT));
	for(int i = 0; i <N; i++){
		free(matrixA[i]) ;
		free(matrixB[i]) ;
		free(matrixC[i]) ;

	}
	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}


