from html import parser

import matplotlib.pyplot as plt
import csv
import argparse
from collections import defaultdict

def load_scaling_data(filename, metric):
    data = defaultdict(lambda: defaultdict(list))
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            binary = row['binary']
            size = int(row['size'])
            p = int(row['procs'])
            val = float(row[metric])
            data[binary][size].append((p, val))
    return data

parser = argparse.ArgumentParser() 
parser.add_argument('--input', required=True, help='Path to strong_summary CSV file')
args = parser.parse_args()

data = load_scaling_data(args.input, 'speedup')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
all_ps = set()

for ax, (binary, sizes) in zip(axes, data.items()):

    for size, points in sizes.items():
        sorted_points = sorted(points)
        if sorted_points[0][0] != 1:
            sorted_points = [(1, 1.0)] + sorted_points
        
        ps = [p for p, v in sorted_points]
        vals = [v for p, v in sorted_points]
        all_ps.update(ps)

        ax.plot(
            ps,
            vals,
            marker='o',
            linewidth=2,
            label=f'N={size}'
        )

    p_list = sorted(list(all_ps))
    # Ideal speedup line
    ax.plot(p_list, p_list, 'k--', linewidth=2, label='Ideal')

    ax.set_title(binary, fontsize=13)
    ax.set_xlabel('Processors / Threads')
    ax.set_ylabel('Speedup')
    ax.set_xticks(p_list)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Strong Scaling Performance Comparison', fontsize=16)
plt.tight_layout()
plt.show()