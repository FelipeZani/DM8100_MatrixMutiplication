#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "include/matrixTools.h"

#define TILE_SIZE 16

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

int main() {
    double *h_A = (double*)malloc(N * N * sizeof(double));
    double *h_B = (double*)malloc(N * N * sizeof(double));
    double *h_C = (double*)malloc(N * N * sizeof(double));

    srand(time(NULL));

    initLinearMatrix(h_A, h_C, h_B, N, 3, 4);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));
    checkCuda(cudaMalloc(&d_A, N * N * sizeof(double)), "Malloc A");
    checkCuda(cudaMalloc(&d_B, N * N * sizeof(double)), "Malloc B");
    checkCuda(cudaMalloc(&d_C, N * N * sizeof(double)), "Malloc C");

    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    checkCuda(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice), "Copy H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice), "Copy H2D B");

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    double startT = (double)clock() / CLOCKS_PER_SEC;
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double endT = (double)clock() / CLOCKS_PER_SEC;

    cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

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