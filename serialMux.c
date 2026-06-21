#include "include/matrixTools.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMux(double *matrixA, double *matrixB, double *matrixC, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}
void matrixMuxTilingBlock(const int N, double *A, double *B, double *C) {
  const int block_size = 64 / sizeof(double); // 64 = common cache line size

  for (int i0 = 0; i0 < N; i0 += block_size) {
    int imin = i0 + block_size > N ? N : i0 + block_size;
    for (int j0 = 0; j0 < N; j0 += block_size) {
      int jmin = j0 + block_size > N ? N : j0 + block_size;

      for (int k0 = 0; k0 < N; k0 += block_size) {
        int kmin = k0 + block_size > N ? N : k0 + block_size;

        for (int i = i0; i < imin; i++) {

          for (int j = j0; j < jmin; j++) {

            for (int k = k0; k < kmin; k++) {
              C[i * N + j] = C[i * N + j] + A[i * N + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    return 1;
  }

  int N = atoi(argv[1]);
  double *matrixA = malloc(N * N * sizeof(double));
  double *matrixB = malloc(N * N * sizeof(double));
  double *matrixC = malloc(N * N * sizeof(double));
  srand(time(NULL));
  initMatrix(matrixA, matrixB, matrixC, N, 2, 3);
  double startT = omp_get_wtime();
  matrixMux(matrixA, matrixB, matrixC, N);
  double endT = omp_get_wtime();

  printf("Size: %dx%d | Exec Time: %f s\n", N, N, (endT - startT));
  printf("First item of C: %f\n", matrixC[0]);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
