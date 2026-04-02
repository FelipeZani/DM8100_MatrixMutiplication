#include "../include/matrixTools.h"
#include <time.h>
const time_t myseed = 9000;
void printMatrix(int ** mat){

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
		    printf("%d ", mat[i][j]);
		}
		printf("\n");
	}

}
void initMatrix(double ** matrixA,double ** matrixB, double ** matrixC, int a , int b){
	
	for(int i = 0; i <N; i++){
			matrixA[i] = malloc(sizeof(double)*N);
			matrixB[i] = malloc(sizeof(double)*N);
			matrixC[i] = malloc(sizeof(double)*N);

	}
	for(int i = 0; i <N; i++){
		for(int j = 0; j <N; j++){
			matrixA[i][j] = 2.0*i+N;
			matrixB[i][j] = 3.0*i+N;

			// matrixA[i][j] =  rand()%((b+1)-a) + a;
			// matrixB[i][j] = rand()%((b+1)-a) + a;
		}
	}	
	
}


