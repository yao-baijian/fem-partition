import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def configure_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 120

def read_data(csv_path):
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['instance'], row['q'])
            data[key]['coarsen_to'].append(int(row['coarsen_to']))
            data[key]['cut_value'].append(float(row['cut_value']))
            data[key]['total_time_s'].append(float(row['total_time_s']))
    return data

def plot_sensitivity(data, out_dir):
    configure_style()
    for (ins, q), vals in data.items():
        # Sort by coarsen_to
        idx = np.argsort(vals['coarsen_to'])
        x = np.array(vals['coarsen_to'])[idx]
        y_cut = np.array(vals['cut_value'])[idx]
        y_time = np.array(vals['total_time_s'])[idx]

        fig, ax1 = plt.subplots(figsize=(6, 4)) # Single column width

        color_cut = '#4c72b0'
        color_time = '#dd8452'

        ax1.set_xlabel('Coarsen Target Nodes')
        ax1.set_ylabel('Cut Value', color=color_cut)
        line1, = ax1.plot(x, y_cut, marker='o', color=color_cut, label='Cut Value', linewidth=1.5, markersize=5)
        ax1.tick_params(axis='y', labelcolor=color_cut)
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Runtime (s)', color=color_time)
        line2, = ax2.plot(x, y_time, marker='s', color=color_time, label='Runtime', linewidth=1.5, markersize=5)
        ax2.tick_params(axis='y', labelcolor=color_time)

        # Merge legends
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', frameon=True)

        out_name = f'sensitivity_{ins}_q{q}.png'
        ax1.set_title(out_name)
        
        fig.tight_layout()
        plt.savefig(out_dir / out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_dir / out_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='build', help='Input CSV directory or file')
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob('bmincut_cfrk_sensitivity_*.csv'))
    else:
        csv_files = [input_path]

    if not csv_files:
        print("No matching CSV files found.")
        return

    out_dir = Path('build')
    out_dir.mkdir(parents=True, exist_ok=True)

    for cf in csv_files:
        print(f"Processing {cf}...")
        data = read_data(cf)
        plot_sensitivity(data, out_dir)

if __name__ == '__main__':
    main()
