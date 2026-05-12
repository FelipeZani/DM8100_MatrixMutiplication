#ifndef MATRIXTOOLS_H
#define MATRIXTOOLS_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

const int N = 500;
extern const time_t myseed;

void printMatrix(double * mat, int N);
void initMatrix(double * matrixA, double * matrixB, double * matrixC, int N, int a , int b);
void initLinearMatrix(double * a, double * c, double * b, int N, int inta , int intb);
double check_norm(double * C_serial, double * C_parallel, int N);

#ifdef __cplusplus
}
#endif

#endif
