#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "include/matrixTools.h"

#define TILE_SIZE 16
// Run with: ./cudaMux 


/* Function that could catch gpu failures early 
   **InProgress** 
*/
static void checkCuda(cudaError_t err, const char *message) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixMulTiled(const double *A, const double *B, double *C, int n) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    double sum = 0.0;

    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        int aCol = tile + threadIdx.x; // global column index of MatrixA
        int bRow = tile + threadIdx.y; // global row index of MatrixB

        if (row < n && aCol < n) {
            sA[threadIdx.y][threadIdx.x] = A[row * n + aCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (bRow < n && col < n) {
            sB[threadIdx.y][threadIdx.x] = B[bRow * n + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) return 1;

    double *h_A = (double*)malloc(n * n * sizeof(double));
    double *h_B = (double*)malloc(n * n * sizeof(double));
    double *h_C = (double*)malloc(n * n * sizeof(double));

    srand(time(NULL));

    initLinearMatrix(h_A, h_C, h_B, n, 3, 4);

    double *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, n * n * sizeof(double)), "Malloc A");
    checkCuda(cudaMalloc(&d_B, n * n * sizeof(double)), "Malloc B");
    checkCuda(cudaMalloc(&d_C, n * n * sizeof(double)), "Malloc C");

    checkCuda(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice), "Copy H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, n * n * sizeof(double), cudaMemcpyHostToDevice), "Copy H2D B");

    dim3 blockDim(16, 16);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    double startT = (double)clock() / CLOCKS_PER_SEC;
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    double endT = (double)clock() / CLOCKS_PER_SEC;

    cudaMemcpy(h_C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Exec Time: %f\n", (endT - startT));
    printf("First item of C %f\n", h_C[0]);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}