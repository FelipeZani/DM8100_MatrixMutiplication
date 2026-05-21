#!/usr/bin/env python3
"""Plot experiment results from saved CSV data.

Usage:
  python analysis/plot_results.py --input analysis/results/raw_results_20260519_123945.csv

This script produces time and speedup plots for each matrix size,
comparing binaries across processor/thread counts.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib is required to plot results. Install it with pip install matplotlib')
    sys.exit(1)


def load_raw_results(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('parsed_time') is None or row.get('parsed_time') == '':
                continue
            try:
                row['parsed_time'] = float(row['parsed_time'])
                row['size'] = int(row['size'])
                row['procs'] = int(row['procs'])
                row['run'] = int(row['run']) if row.get('run') not in (None, '') else None
            except ValueError:
                continue
            rows.append(row)
    return rows


def summarize(rows):
    summary = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row['binary'], row['size'], row['procs'])
        summary[key]['times'].append(row['parsed_time'])
    stats = {}
    for key, data in summary.items():
        times = data['times']
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance ** 0.5
        stats[key] = {
            'mean': mean,
            'std': std,
            'count': len(times),
            'min': min(times),
            'max': max(times),
        }
    return stats


def plot_time_vs_procs(stats, binaries, sizes, procs_list, output_dir):
    out_files = []
    for size in sorted(sizes):
        plt.figure(figsize=(10, 6))
        for binary in sorted(binaries):
            xs = []
            ys = []
            yerr = []
            for p in sorted(procs_list):
                key = (binary, size, p)
                if key not in stats:
                    continue
                xs.append(p)
                ys.append(stats[key]['mean'])
                yerr.append(stats[key]['std'])
            if not xs:
                continue
            plt.errorbar(xs, ys, yerr=yerr, marker='o', label=binary, capsize=3)
        plt.xlabel('Procs / threads')
        plt.ylabel('Execution time (s)')
        plt.title(f'Execution time vs procs, size={size}')
        plt.xscale('log', base=2)
        plt.xticks(sorted(procs_list), sorted(procs_list))
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend()
        filename = os.path.join(output_dir, f'time_vs_procs_size_{size}.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        out_files.append(filename)
    return out_files


def plot_speedup_vs_procs(stats, binaries, sizes, procs_list, output_dir):
    out_files = []
    for size in sorted(sizes):
        plt.figure(figsize=(10, 6))
        any_plot = False
        for binary in sorted(binaries):
            baseline_key = (binary, size, 1)
            if baseline_key not in stats:
                continue
            baseline = stats[baseline_key]['mean']
            xs = []
            ys = []
            for p in sorted(procs_list):
                key = (binary, size, p)
                if key not in stats:
                    continue
                xs.append(p)
                ys.append(baseline / stats[key]['mean'] if stats[key]['mean'] > 0 else None)
            if not xs:
                continue
            any_plot = True
            plt.plot(xs, ys, marker='o', label=binary)
        if not any_plot:
            plt.close()
            continue
        plt.xlabel('Procs / threads')
        plt.ylabel('Speedup (T1 / Tp)')
        plt.title(f'Strong scaling speedup, size={size}')
        plt.xscale('log', base=2)
        plt.xticks(sorted(procs_list), sorted(procs_list))
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend()
        filename = os.path.join(output_dir, f'speedup_vs_procs_size_{size}.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        out_files.append(filename)
    return out_files


def plot_raw_scatter(rows, output_dir):
    out_files = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['binary'], row['size'])].append(row)
    for (binary, size), runs in grouped.items():
        xs = [r['procs'] for r in runs]
        ys = [r['parsed_time'] for r in runs]
        plt.figure(figsize=(8, 5))
        plt.scatter(xs, ys, alpha=0.7)
        plt.xlabel('Procs / threads')
        plt.ylabel('Execution time (s)')
        plt.title(f'Run-level times for {binary} size={size}')
        plt.xscale('log', base=2)
        plt.grid(True, linestyle='--', alpha=0.4)
        filename = os.path.join(output_dir, f'raw_scatter_{binary}_size_{size}.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        out_files.append(filename)
    return out_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw_results CSV')
    parser.add_argument('--output-dir', default='analysis/results/plots', help='Directory to save plots')
    args = parser.parse_args()

    rows = load_raw_results(args.input)
    if not rows:
        print('No valid rows found in', args.input)
        return

    stats = summarize(rows)
    binaries = sorted({row['binary'] for row in rows})
    sizes = sorted({row['size'] for row in rows})
    procs_list = sorted({row['procs'] for row in rows})

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    outputs += plot_time_vs_procs(stats, binaries, sizes, procs_list, args.output_dir)
    outputs += plot_speedup_vs_procs(stats, binaries, sizes, procs_list, args.output_dir)
    outputs += plot_raw_scatter(rows, args.output_dir)

    print('Plots written:')
    for path in outputs:
        print('  ', path)


if __name__ == '__main__':
    main()
