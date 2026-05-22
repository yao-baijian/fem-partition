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
num_steps = 50
dev = 'cpu'
instance = '../partition/full_benchmark_set/as-caida.mtx.hgr'

# ==========================================
# Select the partition method to run:
# 'direct_fem'             : Original FEM applied directly to the clique-expanded hypergraph
# 'coarsen_fem_refine_kahypar' : QUBO-based matching coarsening (FEM) + KaHyPar on coarse hypergraph
# 'coarsen_kahypar_refine' : Multi-level coarsening + KaHyPar initial guess + Greedy refinement
# 'kahyper_like'           : Self-implemented KaHyPar-like coarsening + greedy coarse solve + greedy refinement
# 'pubo_direct'            : Full PUBO-based objective directly on hypergraph (Auto Grad + Opt)
# 'pubo_coarsen'           : Coarsening framework + PUBO on the compressed hyperedges
# 'pubo_q4_explicit'       : Coarsening + explicit formulation via expected_hyperbmincut_explicit
# 'pubo_implicit'         : Coarsening + approximate formulation via expected_hyperbmincut
# ==========================================
partition_method = 'kahyper_like'

# remember requested mode to decide whether to run KaHyPar-based refinement later
requested_method = partition_method

# Default number of partitions. Allow PUBO flows to use q=4 when requested.
if partition_method in ('pubo_q4_explicit', 'pubo_implicit', 'pubo_direct', 'pubo_coarsen'):
    q_ways = 4
else:
    q_ways = 2


def build_coarse_hyperedges(hyperedges_list, original_to_coarse_map, node_count):
    coarse_hyperedges_list = []
    for he in hyperedges_list:
        coarse_he = list(set(int(original_to_coarse_map[v]) for v in he if v < node_count))
        if len(coarse_he) > 1:
            coarse_hyperedges_list.append(coarse_he)
    return coarse_hyperedges_list


def make_q4_pubo_object(hyperedges_list, node_weights_list, cut_func, num_nodes_local, q_local, imbalance_weight=5.0):
    from FEM.problem import weighted_imbalance_penalty

    class _Q4PUBO:
        def __init__(self):
            self.hyperedges = hyperedges_list
            self.node_weights = torch.tensor(node_weights_list, dtype=torch.float32)
            self.imbalance_weight = imbalance_weight

        def expectation(self, _, p):
            self.node_weights = self.node_weights.to(p.device)
            cut_loss = cut_func(None, p, self.hyperedges)
            imb_penalty = weighted_imbalance_penalty(p, self.node_weights.cpu().numpy())
            return cut_loss + self.imbalance_weight * imb_penalty

        def inference(self, _, p):
            q = q_local
            n = num_nodes_local

            if p.dim() == 2:
                if p.shape[1] == q:
                    if p.shape[0] % n != 0:
                        raise ValueError(f"Cannot reshape 2D p with shape {tuple(p.shape)} into (-1, {n}, {q})")
                    p = p.reshape(-1, n, q)
                elif p.shape[0] == n and q == 2:
                    p = p.reshape(1, n, q)
                else:
                    raise ValueError(f"Unexpected 2D p shape: {tuple(p.shape)} for n={n}, q={q}")

            if p.dim() != 3:
                raise ValueError(f"Unexpected p dim: {p.dim()} with shape {tuple(p.shape)}")

            config = torch.zeros_like(p)
            config.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
            return config, torch.zeros(config.shape[0], device=p.device)

    return _Q4PUBO()


def run_kahypar_like_multilevel(clique_graph_local, hyperedges_local, num_nodes_local, q_local, coarsen_to=500):
    coarse_graph_local, coarse_node_weights_local, coarse_groups_local, original_to_coarse_local = coarsen_graph_by_matching(
        clique_graph_local,
        node_weights=torch.ones(num_nodes_local, dtype=torch.float32),
        coarsen_to=coarsen_to,
    )

    coarse_hyperedges_local = build_coarse_hyperedges(hyperedges_local, original_to_coarse_local, num_nodes_local)
    initial_assignment_local = greedy_initial_hypergraph_partition(
        coarse_hyperedges_local,
        coarse_graph_local.shape[0],
        q_local,
        hyperedge_weights=[1.0] * len(coarse_hyperedges_local),
        max_imbalance=0.05,
    )
    initial_assignment_local = greedy_refine_hypergraph(
        initial_assignment_local,
        coarse_hyperedges_local,
        [1.0] * len(coarse_hyperedges_local),
        q=q_local,
        max_passes=5,
        max_imbalance=0.05,
    )

    return coarse_graph_local, coarse_node_weights_local, coarse_groups_local, original_to_coarse_local, initial_assignment_local

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

elif partition_method in ['coarsen_fem_refine_kahypar', 'coarsen_kahypar_refine', 'kahyper_like', 'pubo_coarsen', 'pubo_q4_explicit', 'pubo_implicit']:
    print(f"====== Running {partition_method} ======")
    if partition_method in ['coarsen_kahypar_refine', 'kahyper_like']:
        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, initial_assignment = run_kahypar_like_multilevel(
            clique_graph,
            hyperedges,
            num_nodes,
            q_ways,
        )
        num_coarse_nodes = coarse_graph.shape[0]
        print(f"KaHyPar-like coarse partitioning took: {time.time() - start_time:.4f} seconds")
    else:
        # Step 1: Multi-level coarsening
        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
            clique_graph,
            node_weights=torch.ones(num_nodes, dtype=torch.float32),
            coarsen_to=500,
        )
        
        num_coarse_nodes = coarse_graph.shape[0]
    
    # Step 2: Initial Partition on Coarsened Graph
    # -----------------------------
    # Option A: KaHyPar (Environment Check)
    use_kahypar_refine = False
    if partition_method == 'coarsen_kahypar_refine' and HAS_KAHYPAR:
        print("Using KaHyPar for refinement when available (will use FEM for coarse initial assignment).")
        use_kahypar_refine = True
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

    if partition_method in ['coarsen_kahypar_refine', 'kahyper_like']:
        print("Using self-implemented KaHyPar-like coarse solve (FEM) on the coarsened graph...")
        # `initial_assignment` is already produced by run_kahypar_like_multilevel().
        pass

    # -----------------------------
    # Option D: Explicit q=4 Formulations PUBO on Coarse Graph
    if partition_method == 'pubo_q4_explicit':
        print("Using Explicit q=4 PUBO on the coarsened graph...")
        coarse_hyperedges = build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)

        from FEM.customized_problem.hyper_bmincut import expected_hyperbmincut_explicit

        pubo_obj = make_q4_pubo_object(
            coarse_hyperedges,
            coarse_node_weights,
            expected_hyperbmincut_explicit,
            num_coarse_nodes,
            q_ways,
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
        print(f"Explicit q=4 PUBO partitioning took: {time.time() - start_time:.4f} seconds")

    # -----------------------------
    # Option E: Implicit q=4 Formulation PUBO on Coarse Graph
    if partition_method == 'pubo_implicit':
        print("Using implicit q=4 PUBO on the coarsened graph...")
        from FEM.customized_problem.hyper_bmincut import expected_hyperbmincut
        coarse_hyperedges = build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)

        pubo_obj = make_q4_pubo_object(
            coarse_hyperedges,
            coarse_node_weights,
            expected_hyperbmincut,
            num_coarse_nodes,
            q_ways,
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
        print(f"Implicit q=4 PUBO partitioning took: {time.time() - start_time:.4f} seconds")
        
    # -----------------------------
    # Option B: QUBO-based matching coarsening using FEM, then KaHyPar on coarse hypergraph
    if partition_method == 'coarsen_fem_refine_kahypar':
        print("Using QUBO matching coarsening (FEM) then KaHyPar on coarse hypergraph...")
        from itertools import combinations
        from FEM.cyclic_expansion import solve_qubo_with_fem

        # Build derived graph G' where w'_uv = sum_{e contains u,v} w_e
        wprime = {}
        for he in hyperedges:
            w_e = 1.0
            verts = sorted(set(he))
            for u, v in combinations(verts, 2):
                key = (int(u), int(v))
                wprime[key] = wprime.get(key, 0.0) + w_e

        candidate_edges = list(wprime.keys())
        s = len(candidate_edges)

        # Build QUBO matrix: minimize H = -sum w x + P * sum_v (sum_{i in I(v)} x_i - 1)^2
        import numpy as _np
        Q = _np.zeros((s, s), dtype=float)
        w_vec = _np.array([wprime[e] for e in candidate_edges], dtype=float)
        for i in range(s):
            Q[i, i] -= w_vec[i]

        P = max(1.0, float(w_vec.sum())) * 10.0

        # incidence: vertex -> list of variable indices touching it
        incid = {v: [] for v in range(num_nodes)}
        for idx, (u, v) in enumerate(candidate_edges):
            incid[u].append(idx)
            incid[v].append(idx)

        for v, idxs in incid.items():
            # x_i^2 contributions
            for i in idxs:
                Q[i, i] += P
            # cross terms
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = idxs[i]
                    b = idxs[j]
                    Q[a, b] += 2.0 * P
                    Q[b, a] += 2.0 * P
            # linear -2P x_i
            for i in idxs:
                Q[i, i] += -2.0 * P

        # Solve QUBO with FEM
        assign = solve_qubo_with_fem(Q, num_trials=max(1, num_trials), num_steps=max(10, num_steps), dev=dev)

        # build coarse groups from selected matching edges (greedy to enforce matching)
        matched = set()
        coarse_groups = []
        original_to_coarse = np.full(num_nodes, -1, dtype=np.int64)
        next_c = 0
        for idx, val in enumerate(assign):
            if int(val) == 1:
                u, v = candidate_edges[idx]
                if (u in matched) or (v in matched):
                    continue
                matched.add(u)
                matched.add(v)
                coarse_groups.append([u, v])
                original_to_coarse[u] = next_c
                original_to_coarse[v] = next_c
                next_c += 1

        for v in range(num_nodes):
            if original_to_coarse[v] == -1:
                coarse_groups.append([v])
                original_to_coarse[v] = next_c
                next_c += 1

        num_coarse_nodes = len(coarse_groups)

        # Map hyperedges to coarse hyperedges
        coarse_hyperedges = []
        for he in hyperedges:
            che = list(set(int(original_to_coarse[v]) for v in he if v < num_nodes))
            if len(che) > 1:
                coarse_hyperedges.append(che)

        # Run KaHyPar on coarse hypergraph
        if not HAS_KAHYPAR:
            raise ImportError('kahypar is required for coarsen_fem_refine_kahypar')

        hyperedge_indices = []
        hyperedge_indices_ptrs = [0]
        for he in coarse_hyperedges:
            hyperedge_indices.extend(he)
            hyperedge_indices_ptrs.append(len(hyperedge_indices))

        hypergraph = kahypar.Hypergraph(num_coarse_nodes, len(coarse_hyperedges), hyperedge_indices, hyperedge_indices_ptrs, q_ways, [1]*len(coarse_hyperedges), [1]*num_coarse_nodes)
        context = kahypar.Context()
        try:
            context.loadINIconfiguration('kahypar_config.ini')
        except Exception:
            pass
        context.setK(q_ways)
        context.setEpsilon(0.05)
        kahypar.partition(hypergraph, context)
        coarse_parts = [hypergraph.blockID(i) for i in range(num_coarse_nodes)]
        initial_assignment = expand_coarse_labels(coarse_groups, np.array(coarse_parts), num_nodes)
        print(f"Coarse QUBO matching + KaHyPar partitioning took: {time.time() - start_time:.4f} seconds")

else:
    raise ValueError(f"Unknown partition method: {partition_method}")

# 3. Projection & Refinement
if partition_method in ['coarsen_fem_refine_kahypar', 'coarsen_kahypar_refine', 'kahyper_like', 'pubo_coarsen', 'pubo_q4_explicit', 'pubo_implicit']:
    print("Step 3: Uncoarsening (Projection) back to original hypergraph...")
    group_assignment = expand_coarse_labels(coarse_groups, initial_assignment, num_nodes)
    
    # If KaHyPar was requested and is available, use it to refine the partition
    if requested_method == 'coarsen_kahypar_refine' and use_kahypar_refine:
        print("Step 3: Running KaHyPar refinement on the original hypergraph...")
        # Build hypergraph for kahypar
        hyperedges_indices = []
        hyperedges_ptrs = [0]
        for he in hyperedges:
            hyperedges_indices.extend(he)
            hyperedges_ptrs.append(len(hyperedges_indices))

        hg = kahypar.Hypergraph(num_nodes, len(hyperedges), hyperedges_indices, hyperedges_ptrs, q_ways, [1]*len(hyperedges), [1]*num_nodes)
        for i in range(num_nodes):
            hg.setNodePart(i, int(group_assignment[i]))

        ctx = kahypar.Context()
        try:
            ctx.loadINIconfiguration('kahypar_config.ini')
        except Exception:
            pass
        ctx.setK(q_ways)
        ctx.setEpsilon(0.05)

        kahypar.improvePartition(hg, ctx)
        final_assignment = [hg.blockID(i) for i in range(num_nodes)]
    else:
        print("Step 3: Running Greedy Refinement (Local Swap)...")
        final_assignment = greedy_refine_hypergraph(
            group_assignment, 
            hyperedges, 
            [1.0] * len(hyperedges), 
            q=q_ways, 
            max_passes=5,
            max_imbalance=0.05  # target imbalance <= 0.05 as requested
        )
else:
    final_assignment = initial_assignment

# 4. Final Output & Evaluation
fem_cut_value, _ = evaluate_kahypar_cut_value(final_assignment, hyperedges, [1.0] * len(hyperedges))
counts = np.bincount(final_assignment, minlength=q_ways)
ideal = num_nodes / q_ways
max_imbalance = float(np.max(np.abs(counts - ideal) / ideal))

print(f'\n--- Final Results ---')
print(f'Instance: {instance}')
print(f'Method Executed: {partition_method}')
print(f'Cut Value (k-1 metric): {fem_cut_value}')
print(f'Max Imbalance: {max_imbalance:.6f}')
