#include "../include/matrixTools.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void matrixMux_Serial(double *matrixA, double *matrixB, double *matrixC,
                      int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        matrixC[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 2) {
    if (rank == 0)
      printf("Usage: mpirun -n <cores> %s <Taille N>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  int N = atoi(argv[1]);

  if (N % size != 0) {
    if (rank == 0)
      printf("Erreur : N (%d) // (%d).\n", N, size);
    MPI_Finalize();
    return 1;
  }

  int rows_per_proc = N / size;
  int elements_per_proc = rows_per_proc * N;
  double *a = NULL, *c = NULL, *c_serial = NULL;
  double *b = malloc(N * N * sizeof(double));
  double *aa = malloc(elements_per_proc * sizeof(double));
  double *cc = malloc(elements_per_proc * sizeof(double));
  if (rank == 0) {
    a = malloc(N * N * sizeof(double));
    c = malloc(N * N * sizeof(double));
    c_serial = malloc(N * N * sizeof(double));
    initMatrix(a, b, c, N, 2, 3);
    initMatrix(a, b, c_serial, N, 2, 3);
  }
  for (int i = 0; i < elements_per_proc; i++)
    cc[i] = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  MPI_Scatter(a, elements_per_proc, MPI_DOUBLE, aa, elements_per_proc,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < rows_per_proc; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        cc[i * N + j] += aa[i * N + k] * b[k * N + j];
      }
    }
  }

  MPI_Gather(cc, elements_per_proc, MPI_DOUBLE, c, elements_per_proc,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  if (rank == 0) {
    printf("MPI | Size : %dx%d | Processus: %d | Exec Time: %f s\n", N, N, size,
           end_time - start_time);

    matrixMux_Serial(a, b, c_serial, N);

    double diff = check_norm(c_serial, c, N);
    if (diff == 0.0) {
      printf("Pass (Difference = %f)\n", diff);
    } else {
      printf("Fail (Difference = %f)\n", diff);
    }

    free(a);
    free(c);
    free(c_serial);
  }
  free(b);
  free(aa);
  free(cc);

  MPI_Finalize();
  return 0;
}
