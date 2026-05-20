import sys
sys.path.append('.')
from FEM import FEM
import torch
import time
import numpy as np
import warnings

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    warnings.warn("pymetis is not installed. METIS mode will fail.")

num_trials = 1
num_steps = 100
dev = 'cpu'

# ==========================================
# Select the partition method to run:
# 'direct_fem' : Original FEM applied directly to normal graph
# 'metis'      : PyMetis graph partitioner
# ==========================================
partition_method = 'metis'

# normal graph instances
instance = 'tests/test_instances/karate.txt'
# instance = '../partition/data/ash219/ash219.mtx'
case_type = 'bmincut'
q = 2  # Number of partitions

print(f"Loading {instance}...")
# Use FEM parser to easily load the normal graph
case_bmincut = FEM.from_file(case_type, instance, index_start=1)

if partition_method == 'direct_fem':
    print("====== Running Direct FEM ======")
    start_time = time.time()
    
    case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=q)
    config, result = case_bmincut.solve()
    
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    best_config = config[optimal_inds[0]]
    assignment = best_config.argmax(dim=1).cpu().numpy()
    
    # In bmincut mode, result.min() essentially stores the final cut edges / objective
    # Evaluate via FEM's native infer_bmincut explicitly
    from FEM.problem import infer_bmincut
    _, fem_cut_value = infer_bmincut(case_bmincut.J, best_config.unsqueeze(0))
    fem_cut_value = fem_cut_value.item()
    
    print(f"Direct FEM solve took: {time.time() - start_time:.4f} seconds")
    print(f'{instance}, optimal objective value: {result.min()}, actual edge cut: {fem_cut_value}')

elif partition_method == 'metis':
    if not HAS_METIS:
        raise ImportError("pymetis is required for 'metis' partition method")
    print("====== Running METIS ======")
    
    start_time = time.time()
    
    # We construct the adjacency dict/list for pymetis using the FEM sparse tensor
    J = case_bmincut.J
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    
    n = J.shape[0]
    adjacency_list = [[] for _ in range(n)]
    
    indices = J.indices()
    # It's an unweighted graph typically for normal METIS or we can pass xadj/adjncy.
    # pymetis.part_graph accepts adjacency list of lists
    for idx in range(indices.shape[1]):
        r = int(indices[0, idx])
        c = int(indices[1, idx])
        if r != c:  # no self loops
            adjacency_list[r].append(c)

    # metis
    edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list)
    print(f"METIS solve took: {time.time() - start_time:.4f} seconds")
    
    # evaluate METIS assignment cut with FEM traditional bmincut cut
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(parts):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    fem_eval_cut = fem_eval_cut.item()
    
    print(f'{instance}, METIS edgecuts reported: {edgecuts}, Evaluated actual edge cut: {fem_eval_cut}')

else:
    raise ValueError(f"Unknown partition method: {partition_method}")
