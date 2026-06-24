#include "include/matrixTools.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void matrixMux(double *matrixA, double *matrixB, double *matrixC, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}

void matrixMuxIKJ(double *matrixA, double *matrixB, double *matrixC, int N) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}

void multiply_packed_block(const double *restrict a_packed,
                           const double *restrict b_packed, double *restrict C,
                           const int n, const int block_size) {
  a_packed = __builtin_assume_aligned(a_packed, 32);
  b_packed = __builtin_assume_aligned(b_packed, 32);
  C = __builtin_assume_aligned(C, 32);

  for (int c = 0; c < block_size; ++c) {
    for (int k = 0; k < block_size; ++k) {
      const double b_val = b_packed[c * block_size + k];
      for (int r = 0; r < block_size; ++r) {
        C[r + c * n] += a_packed[r + k * block_size] * b_val;
      }
    }
  }
}

void pack_matrix(double *restrict dest, const double *restrict src, const int n,
                 const int block_size) {
  for (int j = 0; j < block_size; j++) {
    for (int i = 0; i < block_size; i++) {
      dest[j * block_size + i] = src[i + j * n];
    }
  }
}

void tiled_packed_extracted_dgemm(const int n, const double *restrict A,
                                  const double *restrict B,
                                  double *restrict C) {
  const int block_size = 64;
  A = __builtin_assume_aligned(A, 32);
  B = __builtin_assume_aligned(B, 32);
  C = __builtin_assume_aligned(C, 32);

  double *a_packed =
      (double *)aligned_alloc(32, block_size * block_size * sizeof(double));
  double *b_packed =
      (double *)aligned_alloc(32, block_size * block_size * sizeof(double));
  for (int br = 0; br < n; br += block_size) {
    for (int bc = 0; bc < n; bc += block_size) {
      for (int bk = 0; bk < n; bk += block_size) {
        pack_matrix(a_packed, &A[br + bk * n], n, block_size);
        pack_matrix(b_packed, &B[bk + bc * n], n, block_size);
        multiply_packed_block(a_packed, b_packed, &C[br + bc * n], n,
                              block_size);
      }
    }
  }

  free(a_packed);
  free(b_packed);
}

void matrixMuxTilingBlockIKJ(const int N, double *A, double *B, double *C) {
  const int block_size = 64; // 64 = common cache line size
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
  double *matrixC = malloc(N * N * sizeof(double));
  double *matrixD = calloc(N * N, sizeof(double));

  initMatrix(matrixA, matrixB, matrixC, N, 2, 3);

  double startT = omp_get_wtime();
  // matrixMuxTilingBlockIKJ(N, matrixA, matrixB, matrixC);
  // tiled_packed_extracted_dgemm(N, matrixA, matrixB, matrixC);
  // matrixMuxIKJ(matrixA, matrixB, matrixC, N);
  double endT = omp_get_wtime();
  matrixMux(matrixA, matrixB, matrixD, N);
  double diff = check_norm(matrixC, matrixD, N);

  printf("SERIAL MUX (IKJ) | Size: %dx%d | Exec Time: %f s\n", N, N,
         (endT - startT));
  if (diff <= 1e-6) {
    printf("Pass (Difference = %f)\n", diff);
  } else {
    printf("Fail (Difference = %f)\n", diff);
    printf("First element of C: %f\n First element of D: %f", matrixC[0],
           matrixD[0]);
  }
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrixD);

  return 0;
}
