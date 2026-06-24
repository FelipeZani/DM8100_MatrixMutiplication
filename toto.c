#include <stdio.h>
#include <stdlib.h>

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
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      dest[j + block_size * i] = src[i * n + j];
    }
  }
}
void tiled_packed_extracted_dgemm(const int n, const int block_size,
                                  const double *restrict A,
                                  const double *restrict B,
                                  double *restrict C) {
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

int main() {

  double A[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double B[9] = {2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double C[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  tiled_packed_extracted_dgemm(3, 3, A, B, C);

  for (int i = 0; i < 9; i++) {
    printf("%f \n", C[i]);
  }

  return 0;
}

//
// #include <stdio.h>
//
// int main() {
//   // The matrix dimension MUST be a multiple of block_size (64) for this
//   // implementation. Let's use 64x64 for a clean, simple validation block
//   test. const int N = 64;
//
//   // Allocate 32-byte aligned memory for our Column-Major matrices
//   double *A = (double *)aligned_alloc(32, N * N * sizeof(double));
//   double *B = (double *)aligned_alloc(32, N * N * sizeof(double));
//   double *C = (double *)aligned_alloc(32, N * N * sizeof(double));
//
//   if (A == NULL || B == NULL || C == NULL) {
//     fprintf(stderr, "Allocation failed!\n");
//     return 1;
//   }
//
//   // Initialize matrices in Column-Major order
//   // A[r + c * N]
//   for (int c = 0; c < N; ++c) {
//     for (int r = 0; r < N; ++r) {
//       A[r + c * N] = r * 2.0 + N; // A = 2 * Identity Matrix
//       B[r + c * N] = 3.0 * r + N; // B = Simple sequential gradient
//       C[r + c * N] = 0.0;         // C = Initialized to 0
//     }
//   }
//
//   printf("Running tiled_packed_extracted_dgemm...\n\n");
//   tiled_packed_extracted_dgemm(N, A, B, C);
//
//   // Verify results: Since A is 2 * Identity, C should equal 2 * B
//   printf("Verification (Showing a 4x4 top-left corner of the matrices):\n");
//   printf("-------------------------------------------------------------\n");
//   printf("%-15s | %-15s | %-15s\n", "Expected (2*B)", "Actual (C)",
//   "Status");
//   printf("-------------------------------------------------------------\n");
//   int success = 1;
//   for (int c = 0; c < 4; ++c) {
//     for (int r = 0; r < 4; ++r) {
//       // Calculate what C[r + c * N] should mathematically be
//       double expected = 0.0;
//       for (int k = 0; k < N; ++k) {
//         expected += A[r + k * N] * B[k + c * N];
//       }
//
//       double actual = C[r + c * N];
//
//       // Use a small epsilon check for floating point comparisons
//       if (abs(expected - actual) > 1e-6) {
//         success = 0;
//       }
//
//       printf("Row %d, Col %d: Expected %-10.1f | Actual %-10.1f | %s\n", r,
//       c,
//              expected, actual, (success) ? "PASS" : "FAIL");
//     }
//   }
//
//   printf("-------------------------------------------------------------\n");
//   if (success) {
//     printf("SUCCESS: Matrix multiplication logic verified perfectly!\n");
//   } else {
//     printf("FAILURE: Numerical mismatch detected.\n");
//   }
//
//   // Clean up
//   free(A);
//   free(B);
//   free(C);
//
//   return 0;
// }
