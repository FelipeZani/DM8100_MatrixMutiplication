#!/usr/bin/env python3
import argparse
import csv
import os
import matplotlib.pyplot as plt

# how to use:
# python analysis/hpc_matrix_benchmarker.py --output-dir results

# INPUTS data from results/strong_openmp.csv, results/strong_mpi.csv, results/weak_openmp.csv, results/weak_mpi.csv, results/cuda_sizes.csv, results/cross_implementation.csv, and results/comparison_4096.csv
# and GENERATED strong_scaling.png, weak_scaling.png, cuda_performance.png, performance


def load_csv(path):
    """Safely loads a CSV file into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def plot_strong_scaling(results_dir):
    """Plots speedup vs core/thread count for OpenMP and MPI."""
    omp = load_csv(os.path.join(results_dir, "strong_openmp.csv"))
    mpi = load_csv(os.path.join(results_dir, "strong_mpi.csv"))
    
    if not omp and not mpi:
        return

    plt.figure(figsize=(10, 6))
    max_p = 1
    
    if omp:
        x = [int(r['threads']) for r in omp]
        y = [float(r['speedup']) for r in omp]
        plt.plot(x, y, 'o-', linewidth=2, label='OpenMP Speedup')
        max_p = max(max_p, max(x))
        
    if mpi:
        x = [int(r['processes']) for r in mpi]
        y = [float(r['speedup']) for r in mpi]
        plt.plot(x, y, 's-', linewidth=2, label='MPI Speedup')
        max_p = max(max_p, max(x))
        
    # Plot the ideal speedup line
    plt.plot([1, max_p], [1, max_p], 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.xlabel('Cores / Threads')
    plt.ylabel('Speedup')
    plt.title(f'Strong Scaling Speedup (Matrix Size: {omp[0]["size"] if omp else mpi[0]["size"]})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "strong_scaling.png"), dpi=300)
    plt.close()

def plot_weak_scaling(results_dir):
    """Plots execution time vs core/thread count for weak scaling."""
    omp = load_csv(os.path.join(results_dir, "weak_openmp.csv"))
    mpi = load_csv(os.path.join(results_dir, "weak_mpi.csv"))
    
    if not omp and not mpi:
        return

    plt.figure(figsize=(10, 6))
    if omp:
        x = [int(r['threads']) for r in omp]
        y = [float(r['avg_time']) for r in omp]
        plt.plot(x, y, 'o-', linewidth=2, label='OpenMP')
    if mpi:
        x = [int(r['processes']) for r in mpi]
        y = [float(r['avg_time']) for r in mpi]
        plt.plot(x, y, 's-', linewidth=2, label='MPI')
        
    plt.xlabel('Cores / Threads')
    plt.ylabel('Execution Time (s)')
    plt.title('Weak Scaling (Constant Work per Core)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "weak_scaling.png"), dpi=300)
    plt.close()

def plot_cuda_performance(results_dir):
    """Plots execution time for different matrix sizes in CUDA."""
    cuda = load_csv(os.path.join(results_dir, "cuda_sizes.csv"))
    if not cuda:
        print("No CUDA data found for plotting.")
        return

    plt.figure(figsize=(10, 6))
    x = [int(r['size']) for r in cuda]
    y = [float(r['avg_time']) for r in cuda]
    yerr = [float(r['stddev']) for r in cuda]
    
    plt.errorbar(x, y, yerr=yerr, fmt='D-', color='green', linewidth=2, label='CUDA Kernel', capsize=5)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Execution Time (s)')
    plt.title('CUDA Matrix Multiplication Performance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cuda_performance.png"), dpi=300)
    plt.close()

def plot_performance_scaling(results_dir):
    """Plots time vs N for all implementations on a log-log scale."""
    data = load_csv(os.path.join(results_dir, "cross_implementation.csv"))
    if not data:
        print("No cross-implementation data found for plotting.")
        return

    plt.figure(figsize=(10, 6))
    implementations = sorted(list(set(r['implementation'] for r in data)))
    
    for impl in implementations:
        subset = [r for r in data if r['implementation'] == impl]
        x = [int(r['size']) for r in subset]
        y = [float(r['avg_time']) for r in subset]
        yerr = [float(r['stddev']) for r in subset]
        plt.errorbar(x, y, yerr=yerr, fmt='o-', label=impl, capsize=5, linewidth=2)

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Scaling: CPU vs GPU')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "performance_scaling.png"), dpi=300)
    plt.close()

def plot_final_comparison(results_dir):
    """Generates a bar chart comparing all implementations at N=4096."""
    comp = load_csv(os.path.join(results_dir, "comparison_4096.csv"))
    if not comp:
        return

    plt.figure(figsize=(10, 6))
    labels = [r['implementation'] for r in comp]
    times = [float(r['time']) for r in comp]
    stds = [float(r.get('stddev', 0)) for r in comp]
    
    colors = ['#7f7f7f', '#1f77b4', '#d62728', '#2ca02c']
    plt.bar(labels, times, yerr=stds, color=colors[:len(labels)], capsize=10)
    plt.ylabel('Execution Time (s)')
    plt.title('Final Implementation Comparison (N=4096)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scaling experiment results.')
    parser.add_argument('--dir', default='/results', help='Directory containing the CSV results')
    args = parser.parse_args()
    # plot_strong_scaling(args.dir)
    plot_weak_scaling(args.dir)
    # plot_cuda_performance(args.dir)
    # plot_final_comparison(args.dir)
    print(f"Plots generated in: {args.dir}")