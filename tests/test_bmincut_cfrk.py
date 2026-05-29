import sys
sys.path.append('.')
sys.path.append('tests')
from FEM import FEM
import torch
import time
import numpy as np
import warnings
import os
import csv
from datetime import datetime
from utils import simple_kaffpa, coarsen_graph_by_matching, expand_coarse_labels
from FEM.problem import infer_bmincut

num_trials = 1
num_steps = 500
dev = 'cpu'
runs_per_method = 1

# normal graph instances
instances = [
    'tests/test_instances/G1.txt',
]
case_type = 'bmincut'
q_values = [8]  
coarsen_to_values = [100, 200, 500, 1000, 2000]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
build_dir = 'build'
os.makedirs(build_dir, exist_ok=True)
csv_path = os.path.join(build_dir, f'bmincut_cfrk_sensitivity_{timestamp}.csv')
fieldnames = [
    'instance',
    'q',
    'coarsen_to',
    'cut_value',
    'imbalance',
    'total_time_s',
    'coarsen_time_s',
    'init_partition_time_s',
    'refine_time_s',
]
best_rows = []

for instance in instances:
    case_bmincut = FEM.from_file(case_type, instance, index_start=1)
    for q in q_values:
        col_w = (24, 4, 10, 10, 12, 10)
        header_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}} {{:>{col_w[5]}}}"
        sep = ' '.join(['-' * w for w in col_w])
        print(f"\nEvaluating sensitivity for {os.path.basename(instance)}, q={q}")
        print(header_fmt.format('instance', 'q', 'coarsen', 'time(s)', 'cut', 'imbalance'))
        print(sep)

        for coarsen_to in coarsen_to_values:
            best_row = None
            for run_idx in range(runs_per_method):
                start_time = time.perf_counter()
                
                J = case_bmincut.problem.coupling_matrix
                if not J.is_sparse: J = J.to_sparse()
                J = J.coalesce()
                n = J.shape[0]

                # 1. Coarsening
                coarsen_start = time.perf_counter()
                coarse_graph, coarse_node_weights, coarse_groups, _ = coarsen_graph_by_matching(
                    J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
                )
                coarsen_time_s = time.perf_counter() - coarsen_start
                num_coarse_nodes = coarse_graph.shape[0]

                # 2. FEM Initial Partition
                from FEM.initial_partition import fem_initial_partition_kway
                init_start = time.perf_counter()
                try:
                    try: coarse_adj_np = coarse_graph.to_dense().cpu().numpy()
                    except: coarse_adj_np = coarse_graph.cpu().numpy()
                    c_np = coarse_node_weights.cpu().numpy().reshape(-1)
                    coarse_assignment = fem_initial_partition_kway(
                        coarse_adj_np, None, None, c_np, k=q, 
                        lambda_penalty=1.0, num_trials=num_trials, num_steps=num_steps, dev=dev
                    )
                except: raise
                init_partition_time_s = time.perf_counter() - init_start

                initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

                # 3. KaFFPa Refinement
                refine_start = time.perf_counter()
                adjacency_list = [[] for _ in range(n)]
                indices = J.indices()
                for idx in range(indices.shape[1]):
                    r, c = int(indices[0, idx]), int(indices[1, idx])
                    if r != c: adjacency_list[r].append(c)
                xadj, adjncy = [0], []
                for r in range(n):
                    adjncy.extend(adjacency_list[r])
                    xadj.append(len(adjncy))
                vwgt, adjcwgt = [1] * n, [1] * len(adjncy)
                edgecut, part = simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, part=initial_assignment.tolist(), max_passes=10)
                refine_time_s = time.perf_counter() - refine_start

                # Evaluation
                p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                for i, p_group in enumerate(part): p[i, p_group] = 1.0
                _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                
                final_assignment = p.argmax(dim=1).cpu().numpy()
                counts = np.bincount(final_assignment, minlength=q)
                imbalance = float(np.max(np.abs(counts - (n/q)) / (n/q)))
                total_time_s = time.perf_counter() - start_time

                row = {
                    'instance': os.path.basename(instance), 'q': q, 'coarsen_to': coarsen_to,
                    'cut_value': float(fem_eval_cut.item()), 'imbalance': imbalance,
                    'total_time_s': total_time_s, 'coarsen_time_s': coarsen_time_s,
                    'init_partition_time_s': init_partition_time_s, 'refine_time_s': refine_time_s
                }

                if best_row is None or row['cut_value'] < best_row['cut_value'] or (row['cut_value'] == best_row['cut_value'] and row['total_time_s'] < best_row['total_time_s']):
                    best_row = row

            best_rows.append(best_row)
            print(header_fmt.format(best_row['instance'], best_row['q'], best_row['coarsen_to'], best_row['total_time_s'], best_row['cut_value'], best_row['imbalance']))

with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(best_rows)
print(f"Results saved to: {csv_path}")
