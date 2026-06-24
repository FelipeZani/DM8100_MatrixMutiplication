#include "include/matrixTools.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMux_Serial(double *matrixA, double *matrixB, double *matrixC,
                      int N) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}

void matrixMuxIKJ_OMP(double *matrixA, double *matrixB, double *matrixC,
                      int N) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}

void parallelTBMUX(const int N, double *A, double *B, double *C) {
  const int BS = 64;
#pragma omp parallel for collapse(2) schedule(static)
  for (int i0 = 0; i0 < N; i0 += BS) {
    for (int j0 = 0; j0 < N; j0 += BS) {

      int i_max = (i0 + BS < N) ? i0 + BS : N;
      int j_max = (j0 + BS < N) ? j0 + BS : N;

      for (int k0 = 0; k0 < N; k0 += BS) {
        int k_max = (k0 + BS < N) ? k0 + BS : N;

        for (int i = i0; i < i_max; i++) {

          for (int j = j0; j < j_max; j++) {

            double sum = C[i * N + j];

            for (int k = k0; k < k_max; k++) {
              sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
          }
        }
      }
    }
  }
}
void matrixMuxTilingBlock(const int N, double *A, double *B, double *C) {

  const int block_size = 64;

  double end;

  double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (int i0 = 0; i0 < N; i0 += block_size) {
    int imin = i0 + block_size > N ? N : i0 + block_size;
    for (int j0 = 0; j0 < N; j0 += block_size) {
      int jmin = j0 + block_size > N ? N : j0 + block_size;

      for (int k0 = 0; k0 < N; k0 += block_size) {
        int kmin = k0 + block_size > N ? N : k0 + block_size;
        for (int i = i0; i < imin; i++) {
          for (int k = k0; k < kmin; k++) {
            for (int j = j0; j < jmin; j++) {

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
  double *matrixC_OMP = malloc(N * N * sizeof(double));
  double *matrixC_Serial = malloc(N * N * sizeof(double));

  srand(time(NULL));

  initMatrix(matrixA, matrixB, matrixC_OMP, N, 2, 3);
  initMatrix(matrixA, matrixB, matrixC_Serial, N, 2, 3);
  double startT = omp_get_wtime();
  matrixMuxIKJ_OMP(matrixA, matrixB, matrixC_OMP, N);
  matrixMuxTilingBlock(N, matrixA, matrixB, matrixC_OMP);
  double endT = omp_get_wtime();

  matrixMux_Serial(matrixA, matrixB, matrixC_Serial, N);

  double diff = check_norm(matrixC_Serial, matrixC_OMP, N);

  printf("OpenMP (IKJ) | Size: %dx%d | Exec Time: %f s\n", N, N,
         (endT - startT));
  if (diff < 1E-6) {
    printf("Pass (Difference = %f)\n", diff);
  } else {
    printf("Fail (Difference = %f)\n", diff);
  }

  free(matrixA);
  free(matrixB);
  free(matrixC_OMP);
  free(matrixC_Serial);
  return 0;
}
