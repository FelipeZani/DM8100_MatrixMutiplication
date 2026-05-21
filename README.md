# Matrix Multiplication Scaling Analysis

This project implements and evaluates the performance of matrix multiplication across four different paradigms: Serial, OpenMP (shared memory), MPI (distributed memory), and CUDA (GPU acceleration). It includes automated tools for strong and weak scaling analysis.

## Prerequisites

Ensure you have the following installed:
- **GCC**: With OpenMP support.
- **OpenMPI**: For distributed computing (`mpicc`, `mpirun`).
- **CUDA Toolkit**: For NVIDIA GPU execution (`nvcc`).
- **Python 3**: With `matplotlib` for plotting results.

## Compilation

The project uses a `Makefile` to manage builds. You can compile specific versions or everything at once.

```bash
# Compile all implementations
make runall

# Individual targets
make serial
make omp
make mpi
make cuda
```

## Manual Execution

Each binary expects the matrix size $N$ as a command-line argument.

*   **Serial**: `./serialMux 1024`
*   **OpenMP**: `OMP_NUM_THREADS=4 ./openMPMux 1024`
*   **MPI**: `mpirun -n 4 ./mpiMux 1024`
*   **CUDA**: `./cudaMux 1024`

## Benchmarking & Scaling Analysis

The core analysis is driven by `hpc_matrix_benchmarker.py` (formerly `experiments.py`). This script automates multiple runs, parses execution times, and calculates Speedup and Efficiency.

### Strong Scaling
Measures how execution time decreases as you increase processors for a **fixed** problem size.
```bash
python analysis/hpc_matrix_benchmarker.py --mode strong --bins serialMux openMPMux mpiMux --sizes 4096 --procs 1 2 4 8 --runs 3
```

### Weak Scaling
Measures how execution time stays constant as you increase both processors and problem size proportionally.
```bash
python analysis/hpc_matrix_benchmarker.py --mode weak --bins openMPMux mpiMux --sizes 1024 2048 4096 --procs 1 2 4 --runs 3
```

### Parameters
- `--bins`: List of binaries to test.
- `--sizes`: Matrix sizes to evaluate.
- `--procs`: Processor/thread counts (e.g., 1, 2, 4, 8).
- `--runs`: Number of repetitions per configuration to average out noise.
- `--output-dir`: Where to save logs, CSV summaries, and JSON data.

## Visualization

Once benchmarking is complete, use the plotting scripts to generate charts from the generated `.json` or `.csv` files.

### Efficiency Plots
Generates a 4-panel plot showing the parallel efficiency of each implementation.
```bash
python analysis/plot_efficiency.py --input analysis/results/results_TIMESTAMP.json
```

### Speedup Plots
Visualizes the speedup achieved compared to the ideal linear scaling.
```bash
python analysis/tmp/plot_4split.py --input analysis/results/strong_summary_TIMESTAMP.csv
```

## Project Structure
- `src/`: Common utilities for matrix initialization and verification.
- `analysis/`: Benchmarking scripts and visualization tools.
- `include/`: Header files.
