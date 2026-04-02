#ifndef MATRIXTOOLS_H
#define MATRIXTOOLS_H
#include <stdio.h>
#include <stdlib.h>




#define N 1000
#define M 3
void printMatrix(int ** mat);

extern const time_t myseed;
void initMatrix(double ** matrixA, double ** matrixB, double ** matrixC, int a , int b);



#endif

