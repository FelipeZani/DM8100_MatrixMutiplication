#include "../include/matrixTools.h"
#include <time.h>
const time_t myseed = 9000;

void printMatrix(double * mat, int N){
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			printf("%f ", mat[i * N + j]);
		}
		printf("\n");
	}
}

void initLinearMatrix(double * a, double * c, double * b, int N, int inta, int intb){
	for(int i=0; i<N; i++) {
		for(int j = 0;j<N;j++ ){
			a[i*N+j] = (double) (rand()%((intb+1)-inta) + inta);
			b[i*N+j] = (double) (rand()%((intb+1)-inta) + inta);
			c[i*N+j] = 0.0;
		}
	}
}

void initMatrix(double * matrixA, double * matrixB, double * matrixC, int N, int a , int b){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			matrixA[i * N + j] = 2.0 * i + N;
			matrixB[i * N + j] = 3.0 * i + N;
			matrixC[i * N + j] = 0.0;
		}
	}
}
double check_norm(double * C_serial, double * C_parallel, int N) {
	double sum = 0.0;
	for(int i = 0; i < N * N; i++) {
		double diff = C_serial[i] - C_parallel[i];
		sum += (diff > 0) ? diff : -diff;
	}
	return sum;
}