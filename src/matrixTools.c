#include "../include/matrixTools.h"
void printMatrix(int ** mat){

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
		    printf("%d ", mat[i][j]);
		}
		printf("\n");
	}

}
void initMatrix(int ** matrixA,int ** matrixB, int ** matrixC, int a , int b){
	
	for(int i = 0; i <N; i++){
			matrixA[i] = malloc(sizeof(int)*N);
			matrixB[i] = malloc(sizeof(int)*N);
			matrixC[i] = malloc(sizeof(int)*N);

	}
	for(int i = 0; i <N; i++){
		for(int j = 0; j <N; j++){
			matrixA[i][j] =  rand()%((b+1)-a) + a;
			matrixB[i][j] = rand()%((b+1)-a) + a;
		}
	}	
	
}


