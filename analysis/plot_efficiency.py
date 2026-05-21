import matplotlib.pyplot as plt
import json
import argparse
from collections import defaultdict

def load_scaling_data(filename):
    data = defaultdict(lambda: defaultdict(list))
    with open(filename, 'r') as f:
        payload = json.load(f)
    
    # Prefer strong_summary as it contains the pre-calculated efficiency metric
    summary = payload.get('strong_summary') or payload.get('weak_summary') or []
    
    for row in summary:
        if 'efficiency' in row and row['efficiency'] is not None:
            binary = row['binary']
            size = int(row['size'])
            p = int(row['procs'])
            # Efficiency in JSON is decimal (0.0-1.0), convert to % for the plot
            val = float(row['efficiency']) * 100
            data[binary][size].append((p, val))
            #print(f"Loaded efficiency data: {binary} | Size: {size} | Procs: {p} | Efficiency: {val:.2f}%")     
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to results JSON file')
args = parser.parse_args()

efficiency_data = load_scaling_data(args.input)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
all_ps = set()

for ax, (binary, sizes) in zip(axes, efficiency_data.items()):

    for size, points in sizes.items():
        sorted_points = sorted(points)
        if sorted_points[0][0] != 1:
            sorted_points = [(1, 100.0)] + sorted_points
        
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
    # Ideal efficiency line
    ax.axhline(
        y=100,
        linestyle='--',
        linewidth=2,
        color='black',
        label='Ideal (100%)'
    )

    ax.set_title(binary, fontsize=13)
    ax.set_xlabel('Processors / Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xticks(p_list)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Strong Scaling Efficiency Comparison', fontsize=16)
plt.tight_layout()
plt.show()