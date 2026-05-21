#!/usr/bin/env python3
"""
Matrix Multiplication Performance Profiler
Evaluates strong and weak scaling for Serial, OpenMP, MPI, and CUDA implementations.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
STRONG_COUNTS = [1, 2, 4, 8, 16]
STRONG_SIZE = 2048
CUDA_SIZES = [1024, 2048, 4096]
ALL_SIZES = [1024, 2048, 4096]
WEAK_BASE_SIZE = 1024

TIME_RE = re.compile(r"([0-9]+\.?[0-9eE+-]*)")

# how to run this: python analysis/experiments2.py --runs 5 --output results


def get_executable(name):
    if os.name == "nt":
        return f"{name}.exe"
    return f"./{name}"


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def weak_size(processes):
    """
    Matrix multiplication:
    work ~ N^3

    Weak scaling:
    N ~ p^(1/3)
    """
    return int(round(WEAK_BASE_SIZE * (processes ** (1.0 / 3.0))))


def parse_time(output):
    # More specific patterns to avoid capturing matrix dimensions or other integers
    patterns = [
        r"Exec Time:\s*([0-9]+\.?[0-9eE+-]*)",
        r"Time taken:\s*([0-9]+\.?[0-9eE+-]*)",
        r"([0-9]+\.?[0-9eE+-]*)\s*seconds"
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
            
    for line in output.splitlines():
        if "time" in line.lower():
            matches = TIME_RE.findall(line)
            if matches:
                try:
                    return float(matches[-1])
                except:
                    pass
    matches = TIME_RE.findall(output)
    if matches:
        try:
            return float(matches[-1])
        except:
            pass

    return None


def run_command(cmd, env=None):
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env=env
    )
    output = result.stdout + result.stderr
    return output


def measure(cmd, runs=3, env=None):

    times = []
    for _ in range(runs):
        output = run_command(cmd, env)
        t = parse_time(output)
        if t is None:
            raise RuntimeError(f"Command '{cmd}' failed to produce a valid 'Exec Time'. Output:\n{output}")
        times.append(t)

    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "times": times,
        "avg": avg,
        "std": std
    }


def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def write_csv(filename, rows):
    if not rows:
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

# --------------------------------------------------
# EXECUTION HELPERS
# --------------------------------------------------

def run_serial(size, runs):
    cmd = f"{get_executable('serialMux')} {size}"

    return measure(cmd, runs)


def run_openmp(size, threads, runs):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = f"{get_executable('openMPMux')} {size}"

    return measure(cmd, runs, env)


def run_mpi(size, processes, runs):
    cmd = (
        f"mpirun -n {processes} "
        f"{get_executable('mpiMux')} {size}"
    )

    return measure(cmd, runs)


def run_cuda(size, runs):
    cmd = f"{get_executable('cudaMux')} {size}"

    return measure(cmd, runs)


# --------------------------------------------------
# STRONG SCALING
# --------------------------------------------------

def strong_openmp(results_dir, runs):

    rows = []
    baseline = run_openmp( STRONG_SIZE, 1, runs )
    t1 = baseline["avg"]
    for threads in STRONG_COUNTS:

        result = run_openmp(
            STRONG_SIZE,
            threads,
            runs
        )
        speedup = t1 / result["avg"]
        efficiency = speedup / threads
        rows.append({
            "implementation": "OpenMP",
            "size": STRONG_SIZE,
            "threads": threads,
            "avg_time": result["avg"],
            "stddev": result["std"],
            "speedup": speedup,
            "efficiency": efficiency
        })

    write_csv( f"{results_dir}/strong_openmp.csv", rows)

def strong_mpi(results_dir, runs):
    rows = []
    baseline = run_mpi(
        STRONG_SIZE,
        1,
        runs
    )

    t1 = baseline["avg"]

    for processes in STRONG_COUNTS:

        result = run_mpi(
            STRONG_SIZE,
            processes,
            runs
        )
        speedup = t1 / result["avg"]
        efficiency = speedup / processes
        rows.append({
            "implementation": "MPI",
            "size": STRONG_SIZE,
            "processes": processes,
            "avg_time": result["avg"],
            "stddev": result["std"],
            "speedup": speedup,
            "efficiency": efficiency
        })

    write_csv( f"{results_dir}/strong_mpi.csv", rows)


# --------------------------------------------------
# WEAK SCALING
# --------------------------------------------------

def weak_openmp(results_dir, runs):

    rows = []
    for threads in STRONG_COUNTS:
        size = weak_size(threads)
        result = run_openmp(
            size,
            threads,
            runs
        )
        rows.append({
            "implementation": "OpenMP",
            "threads": threads,
            "size": size,
            "avg_time": result["avg"],
            "stddev": result["std"]
        })

    write_csv(
        f"{results_dir}/weak_openmp.csv",
        rows
    )


def weak_mpi(results_dir, runs):

    rows = []
    for processes in STRONG_COUNTS:
        size = weak_size(processes)
        result = run_mpi(
            size,
            processes,
            runs
        )
        rows.append({
            "implementation": "MPI",
            "processes": processes,
            "size": size,
            "avg_time": result["avg"],
            "stddev": result["std"]
        })

    write_csv( f"{results_dir}/weak_mpi.csv",  rows)


# --------------------------------------------------
# CUDA STUDY
# --------------------------------------------------

def cuda_study(results_dir, runs):

    rows = []
    for size in CUDA_SIZES:
        result = run_cuda(
            size,
            runs
        )
        rows.append({
            "implementation": "CUDA",
            "size": size,
            "avg_time": result["avg"],
            "stddev": result["std"]
        })
    write_csv( f"{results_dir}/cuda_sizes.csv", rows)


def cross_implementation_study(results_dir, runs):
    """Benchmarks all strategies across different N to show scaling trends."""
    rows = []
    for size in ALL_SIZES:
        print(f"  Benchmarking N={size}...")
        serial = run_serial(size, runs)
        omp = run_openmp(size, 8, runs)
        mpi = run_mpi(size, 8, runs)
        cuda = run_cuda(size, runs)
        
        rows.append({
            "size": size,
            "implementation": "Serial",
            "avg_time": serial["avg"],
            "stddev": serial["std"]
        })
        rows.append({
            "size": size,
            "implementation": "OpenMP (8)",
            "avg_time": omp["avg"],
            "stddev": omp["std"]
        })
        rows.append({
            "size": size,
            "implementation": "MPI (8)",
            "avg_time": mpi["avg"],
            "stddev": mpi["std"]
        })
        rows.append({
            "size": size,
            "implementation": "CUDA",
            "avg_time": cuda["avg"],
            "stddev": cuda["std"]
        })
        
    write_csv(f"{results_dir}/cross_implementation.csv", rows)


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

def final_comparison(results_dir, runs):

    size = 4096
    serial = run_serial(size, runs)
    omp = run_openmp(size, 16, runs)
    mpi = run_mpi(size, 16, runs)
    cuda = run_cuda(size, runs)
    rows = [
        {
            "implementation": "Serial",
            "time": serial["avg"],
            "stddev": serial["std"]
        },
        {
            "implementation": "OpenMP_16",
            "time": omp["avg"],
            "stddev": omp["std"]
        },
        {
            "implementation": "MPI_16",
            "time": mpi["avg"],
            "stddev": mpi["std"]
        },
        {
            "implementation": "CUDA",
            "time": cuda["avg"],
            "stddev": cuda["std"]
        }
    ]
    write_csv( f"{results_dir}/comparison_4096.csv", rows)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--output",
        default="results"
    )
    args = parser.parse_args()
    ensure_dir(args.output)

    print("Running OpenMP strong scaling...")
    strong_openmp(args.output, args.runs)

    print("Running MPI strong scaling...")
    strong_mpi(args.output, args.runs)

    print("Running OpenMP weak scaling...")
    weak_openmp(args.output, args.runs)

    print("Running MPI weak scaling...")
    weak_mpi(args.output, args.runs)

    print("Running CUDA size study...")
    cuda_study(args.output, args.runs)

    print("Running cross-implementation scaling study...")
    cross_implementation_study(args.output, args.runs)

    print("Running final comparison...")
    final_comparison(args.output, args.runs)

    print("Done.")


if __name__ == "__main__":
    main()