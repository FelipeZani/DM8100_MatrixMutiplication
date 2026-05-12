# INTERNAL REPORT                                      

## CUDA
- [x] Offload the computation to a GPU accelerator.
- [ ] Write a CUDA kernel where each thread computes one element of the output matrix.
- [ ] Manage memory transfers between the host and the device.     


## **The Kernel**
`__global__ void matrixMulTiled(const double *A, const double *B, double *C, int n)`
### Thread & Block Mapping 
```cpp
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
```
***Explanation:***   Each thread computes one output cell `C[row][col]`.  \
Blocks are arranged in a 2D grid, each block handles a 16x16 tile of the output matrix.                         
### Shared Memory 
```cpp
__shared__ double sA[TILE_SIZE][TILE_SIZE];
__shared__ double sB[TILE_SIZE][TILE_SIZE];
```
***Explanation:***   `Shared memory` is ~100x faster than global memory (on-chip cache vs. main GPU RAM), \
All 256 threads in a block can access these tiny matrices with minimal latency.\
Instead of hammering global memory 16 times per thread, we load once and reuse
### Tiled Algorithm 
```cpp
for (int tile = 0; tile < n; tile += TILE_SIZE) {
    // Load tile of A and B into shared memory
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
    
    __syncthreads();  // Wait for all threads to load
    
    // Compute partial result using shared memory
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }
    
    __syncthreads();  // Sync before next load
}
```
` tile`: represents the starting index (the offset) of the current tile in the global matrix.

***Explanation:***   Process the multiplication in blocks instead of the entire matrix.

Each tile iteration: 
1. Load 16×16 chunk of A and B into fast shared memory
2. All threads compute their partial products from shared memory (fast)
3. Accumulate into sum 

Repeat for the next tile
#### Final Write Back
```cpp
if (row < n && col < n) {
    C[row * n + col] = sum;
}
```
***Explanation:***   Only valid threads write (handles non-squeare matrices good).