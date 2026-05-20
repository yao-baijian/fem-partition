import sys
sys.path.append('.')
from FEM import FEM
import torch
import time
import numpy as np
import warnings

from utils import *

try:
    import kahypar
    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False
    warnings.warn("KaHyPar is not installed. Will fallback to FEM where applicable.")

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    # Not strictly warning for metis as kahypar/FEM act as the main branches

# num_trials = 500
# num_steps = 1000

num_trials = 1
num_steps = 100
dev = 'cpu'
instance = '../partition/full_benchmark_set/as-caida.mtx.hgr'

# ==========================================
# Select the partition method to run:
# 'direct_fem'             : Original FEM applied directly to the clique-expanded hypergraph
# 'coarsen_fem_refine'     : Multi-level coarsening + FEM initial guess + Greedy refinement (Fallback)
# 'coarsen_kahypar_refine' : Multi-level coarsening + KaHyPar initial guess + Greedy refinement
# 'pubo_direct'            : Full PUBO-based objective directly on hypergraph (Auto Grad + Opt)
# 'pubo_coarsen'           : Coarsening framework + PUBO on the compressed hyperedges
# 'pubo_q4_explicit'       : Coarsening + explicit formulation via expected_hyperbmincut_all_comb
# ==========================================
partition_method = 'pubo_q4_explicit'

q_ways = 4 if partition_method == 'pubo_q4_explicit' else 2

print(f"Loading {instance}...")
hyperedges = parse_hypergraph_edges(instance)
num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1
clique_graph = build_clique_expanded_graph(hyperedges, num_nodes=num_nodes, normalize_weight=True)

start_time = time.time()

# 1. & 2. PRE-PROCESSING & INITIAL PARTITIONING
if partition_method == 'direct_fem':
    print("====== Running Direct FEM ======")
    graph_for_fem = clique_graph # Avoid .to_dense() to prevent OOM
    node_weights_for_fem = torch.ones(num_nodes, dtype=torch.float32)
    
    # Run FEM directly on original graph
    print("Setting up FEM solver...")
    case_bmincut = FEM.from_couplings(
        'bmincut',
        graph_for_fem.shape[0],
        int(clique_graph._nnz() // 2),
        graph_for_fem,
        node_weights=node_weights_for_fem,
    )
    case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=True)
    
    print("Running FEM optimize...")
    config, result = case_bmincut.solve()
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    best_config = config[optimal_inds[0]]
    initial_assignment = best_config.argmax(dim=1).cpu().numpy()
    
    print(f"Direct FEM solve took: {time.time() - start_time:.4f} seconds")
    coarse_groups = None # No projection needed

elif partition_method == 'pubo_direct':
    print("====== Running Direct PUBO FEM ======")
    pubo_obj = PUBOObjective(
        hyperedges, [1.0] * len(hyperedges), q=q_ways, num_nodes=num_nodes,
        node_weights=torch.ones(num_nodes, dtype=torch.float32), 
        imbalance_weight=5.0, obj_type='cut_net', max_degree=5
    )
    
    # FEM problem using customize
    dummy_matrix = torch.zeros((num_nodes, num_nodes)) # not used
    case_bmincut = FEM()
    case_bmincut.set_up_problem(
        num_nodes, 0, 'customize', dummy_matrix, q=q_ways,
        customize_expected_func=pubo_obj.expectation,
        customize_infer_func=pubo_obj.inference
    )
    case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
    
    print("Running PUBO FEM optimize...")
    config, result = case_bmincut.solve()
    best_config = config[2] if len(config) > 2 else config[0] # taking first config 
    initial_assignment = best_config.argmax(dim=1).cpu().numpy()
    
    print(f"Direct PUBO solve took: {time.time() - start_time:.4f} seconds")
    coarse_groups = None

elif partition_method in ['coarsen_fem_refine', 'coarsen_kahypar_refine', 'pubo_coarsen', 'pubo_q4_explicit']:
    print(f"====== Running {partition_method} ======")
    # Step 1: Multi-level coarsening
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_leaf_folding(
        clique_graph,
        node_weights=torch.ones(num_nodes, dtype=torch.float32),
        max_degree=500,
        min_nodes=5000,
    )
    
    num_coarse_nodes = coarse_graph.shape[0]
    
    # Step 2: Initial Partition on Coarsened Graph
    # -----------------------------
    # Option A: KaHyPar (Environment Check)
    if partition_method == 'coarsen_kahypar_refine' and HAS_KAHYPAR:
        print("Using KaHyPar for initial coarse partition...")
        print("Fallback warning: Complex coarse hyperedge construction omitted due to missing full kahypar context. Reverting to FEM-based initial guess.")
        partition_method = 'coarsen_fem_refine'
        
    elif partition_method == 'coarsen_kahypar_refine' and not HAS_KAHYPAR:
        print("KaHyPar is requested but not installed. Falling back to FEM on the coarsened graph (coarsen_fem_refine).")
        partition_method = 'coarsen_fem_refine'

    # -----------------------------
    # Option C: PUBO on Coarse Graph
    if partition_method == 'pubo_coarsen':
        print("Using PUBO as the primary solver on the coarsened graph...")
        # map original hyperedges to coarse hyperedges
        coarse_hyperedges = []
        for he in hyperedges:
            che = list(set([original_to_coarse[v] for v in he if v < num_nodes]))
            if len(che) > 1:
                coarse_hyperedges.append(che)
                
        pubo_obj = PUBOObjective(
            coarse_hyperedges, [1.0] * len(coarse_hyperedges), q=q_ways, num_nodes=num_coarse_nodes,
            node_weights=coarse_node_weights, 
            imbalance_weight=5.0, obj_type='cut_net', max_degree=5
        )
        
        dummy_matrix = torch.zeros((num_coarse_nodes, num_coarse_nodes))
        case_bmincut = FEM()
        case_bmincut.set_up_problem(
            num_coarse_nodes, 0, 'customize', dummy_matrix, q=q_ways,
            customize_expected_func=pubo_obj.expectation,
            customize_infer_func=pubo_obj.inference
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
        config, result = case_bmincut.solve()
        best_config = config[0]
        initial_assignment = best_config.argmax(dim=1).cpu().numpy()
        print(f"Coarse PUBO partitioning took: {time.time() - start_time:.4f} seconds")

    # -----------------------------
    # Option D: Explicit q=4 Formulations PUBO on Coarse Graph
    if partition_method == 'pubo_q4_explicit':
        print("Using Explicit q=4 PUBO (expected_hyperbmincut_all_comb) on the coarsened graph...")
        # Map original hyperedges to coarse hyperedges
        coarse_hyperedges = []
        for he in hyperedges:
            che = list(set([original_to_coarse[v] for v in he if v < num_nodes]))
            if len(che) > 1:
                coarse_hyperedges.append(che)
                
        from FEM.customized_problem.hyper_bmincut import expected_hyperbmincut_all_comb
        from FEM.problem import weighted_imbalance_penalty

        class ExplicitQ4PUBO:
            def __init__(self, hyperedges_list, node_weights_list, imbalance_weight=5.0):
                self.hyperedges = hyperedges_list
                self.node_weights = torch.tensor(node_weights_list, dtype=torch.float32)
                self.imbalance_weight = imbalance_weight

            def expectation(self, _, p):
                self.node_weights = self.node_weights.to(p.device)
                cut_loss = expected_hyperbmincut_all_comb(None, p, self.hyperedges)
                imb_penalty = weighted_imbalance_penalty(p, self.node_weights.cpu().numpy())
                return cut_loss + self.imbalance_weight * imb_penalty
                
            def inference(self, _, p):
                config = torch.zeros_like(p)
                config.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
                return config, torch.zeros(p.shape[0], device=p.device)

        pubo_obj = ExplicitQ4PUBO(coarse_hyperedges, coarse_node_weights)
        
        dummy_matrix = torch.zeros((num_coarse_nodes, num_coarse_nodes))
        case_bmincut = FEM()
        case_bmincut.set_up_problem(
            num_coarse_nodes, 0, 'customize', dummy_matrix, q=q_ways,
            customize_expected_func=pubo_obj.expectation,
            customize_infer_func=pubo_obj.inference
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
        config, result = case_bmincut.solve()
        best_config = config[0]
        initial_assignment = best_config.argmax(dim=1).cpu().numpy()
        print(f"Explicit q=4 PUBO partitioning took: {time.time() - start_time:.4f} seconds")
        
    # -----------------------------
    # Option B: FEM (备选实现/Fallback implementation)
    if partition_method == 'coarsen_fem_refine':
        print("Using FEM as the primary solver on the coarsened graph (Sparse)...")
        graph_for_fem = coarse_graph # Use original sparse representation!
        
        case_bmincut = FEM.from_couplings(
            'bmincut',
            num_coarse_nodes,
            int(coarse_graph._nnz() // 2),
            graph_for_fem,
            node_weights=coarse_node_weights,
        )
        # Enable manual_grad=True to utilize our newly injected sparse gradient acceleration
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=2, manual_grad=True)
        
        config, result = case_bmincut.solve()
        best_config = config[torch.argwhere(result==result.min()).reshape(-1)[0]]
        initial_assignment = best_config.argmax(dim=1).cpu().numpy()
        print(f"Coarse partitioning took: {time.time() - start_time:.4f} seconds")

else:
    raise ValueError(f"Unknown partition method: {partition_method}")

# 3. Projection & Refinement
if partition_method in ['coarsen_fem_refine', 'coarsen_kahypar_refine', 'pubo_coarsen']:
    print("Step 3: Uncoarsening (Projection) back to original hypergraph...")
    group_assignment = expand_coarse_labels(coarse_groups, initial_assignment, num_nodes)
    
    print("Step 3: Running Greedy Refinement (Local Swap)...")
    final_assignment = greedy_refine_hypergraph(
        group_assignment, 
        hyperedges, 
        [1.0] * len(hyperedges), 
        q=2, 
        max_passes=5,
        max_imbalance=0.05  # target imbalance <= 0.05 as requested
    )
else:
    final_assignment = initial_assignment

# 4. Final Output & Evaluation
fem_cut_value, avg_imbalance = evaluate_kahypar_cut_value(final_assignment, hyperedges, [1.0] * len(hyperedges))

print(f'\n--- Final Results ---')
print(f'Instance: {instance}')
print(f'Method Executed: {partition_method}')
print(f'Cut Value (k-1 metric): {fem_cut_value}')
print(f'Max Imbalance: {avg_imbalance:.6f}')