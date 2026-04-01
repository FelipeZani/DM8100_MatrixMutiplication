#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include "include/matrixTools.h"

//Reads memory in a straight line
void matrixMuxIKJ(int ** matrixA, int ** matrixB, int ** matrixC){


	#pragma omp parallel 
	{
		#pragma omp for schedule(dynamic,M)  collapse(2) 
		for(int i = 0; i < N; i++) {
			
			for(int k = 0; k < N; k++) {
					
				for(int j = 0; j < N; j++) {
					matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				}
			}
		}

	}

}
void matrixMuxIJK(int ** matrixA, int ** matrixB, int ** matrixC){
	#pragma omp parallel 
	{
		#pragma omp for schedule(dynamic,M)  collapse(2) 
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				for(int k = 0; k < N; k++) {
					matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				}
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
	initMatrix(matrixA, matrixB,matrixC,2,3);
	double startT = omp_get_wtime();
	double endT;


	matrixMuxIJK(matrixA,matrixB,matrixC);
	// matrixMuxIKJ(matrixA,matrixB,matrixC); //faster than previous implementation
	
	endT = omp_get_wtime();

    // Print result
			
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

//Interisting reading which may be useful:
// https://stackoverflow.com/questions/28482833/understanding-the-collapse-clause-in-openmp#28483812 - Trying to fuse the three loops instead of using a nested one 
