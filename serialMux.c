#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define N 100
void initMatrix(int ** matrixA,int ** matrixB, int ** matrixC, int a , int b){
	
	for(int i = 0; i <N; i++){
			matrixA[i] = malloc(sizeof(int)*N);
			matrixB[i] = malloc(sizeof(int)*N);
			matrixC[i] = malloc(sizeof(int)*N);

	}
	for(int i = 0; i <N; i++){
		for(int j = 0; j <N; j++){
			matrixA[i][j] = a;
			matrixB[i][j] = b;
		}
	}	
	
}
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
	matrixMux(matrixA,matrixB,matrixC);

    // Print result
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
		    printf("%d ", matrixC[i][j]);
		}
		printf("\n");
	}
		
	for(int i = 0; i <N; i++){
		free(matrixA[i]) ;
		free(matrixB[i]) ;

	}


	return 0;
}


