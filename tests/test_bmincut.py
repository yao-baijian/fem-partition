import sys
sys.path.append('.')
sys.path.append('tests')
from FEM import FEM
import torch
import time
import numpy as np
import warnings
import os
from utils import simple_kaffpa

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    warnings.warn("pymetis is not installed. METIS mode will fail.")

num_trials = 1
num_steps = 500
dev = 'cpu'
anneal = 'lin'
manual_grad = False

# ==========================================
# Select the partition method to run:
# 'direct_fem'                : Original FEM applied directly to normal graph
# 'coarse_fem_refine_metis'   : Multi-level coarsening + FEM coarse opt + METIS fine opt
# 'coarse_fem_refine_kahypar' : Multi-level coarsening + FEM coarse opt + KaHyPar fine opt
# 'coarse_fem_refine_kaffpa'  : Multi-level coarsening + FEM coarse opt + KaFFPa fine opt
# 'metis'                     : PyMetis graph partitioner alone
# 'kahypar'                   : KaHyPar partitioner alone
# 'kaffpa'                    : KaFFPa partitioner alone
# ==========================================
partition_methods = ['direct_fem', 'kaffpa', 'coarse_fem_refine_kaffpa']

# normal graph instances
instance = 'tests/test_instances/G1.txt'
# instance = '../partition/data/ash219/ash219.mtx'
case_type = 'bmincut'
q = 4  # Number of partitions

# Use FEM parser to easily load the normal graph
case_bmincut = FEM.from_file(case_type, instance, index_start=1)

# Enable multi-level coarsening for kaffpa (and FEM+KaFFPa uses coarsening
# by design). Set to False to run vanilla KaFFPa on the full graph.
enable_multilevel_coarsen_for_kaffpa = True
coarsen_to = 500

for partition_method in partition_methods:
    p = None
    best_config = None

    # Print table header once before first method (fixed-width columns)
    if partition_method == partition_methods[0]:
        col_w = (24, 22, 10, 12, 10)  # instance, method, time, cut, imbalance
        header_fmt = f"{{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}}"
        sep = ' '.join(['-' * w for w in col_w])
        print(header_fmt.format('instance', 'method', 'time(s)', 'cut', 'imbalance'))
        print(sep)
    # print(f'\n====== Evaluating Method: {partition_method} ======')
    if partition_method == 'direct_fem':
        start_time = time.time()
        
        case_bmincut.set_up_solver(num_trials, num_steps, anneal=anneal, dev=dev, q=q, manual_grad=manual_grad)
        config, result = case_bmincut.solve()
        
        optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
        best_config = config[optimal_inds[0]]
        assignment = best_config.argmax(dim=1).cpu().numpy()
        
        # In bmincut mode, result.min() essentially stores the final cut edges / objective
        # Evaluate via FEM's native infer_bmincut explicitly
        from FEM.problem import infer_bmincut
        _, fem_cut_value = infer_bmincut(case_bmincut.problem.coupling_matrix, best_config.unsqueeze(0))
        fem_cut_value = fem_cut_value.item()
        
        # suppressed intermediate prints; only table row will be output

    elif partition_method == 'metis':
        if not HAS_METIS:
            raise ImportError("pymetis is required for 'metis' partition method")
        start_time = time.time()
        
        # We construct the adjacency dict/list for pymetis using the FEM sparse tensor
        J = case_bmincut.problem.coupling_matrix
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
        # suppressed intermediate prints
        
        # evaluate METIS assignment cut with FEM traditional bmincut cut
        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(parts):
            p[0, i, p_group] = 1.0
            
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        fem_eval_cut = fem_eval_cut.item()
        
        # suppressed intermediate prints

    elif partition_method == 'coarse_fem_refine_metis':
        start_time = time.time()
        
        from utils import coarsen_graph_by_matching, expand_coarse_labels
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]
        
        # 1. Multi-level coarsening
        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
            J,
            node_weights=torch.ones(n, dtype=torch.float32),
            coarsen_to=500,
        )
        
        num_coarse_nodes = coarse_graph.shape[0]
        
        # 2. FEM solver on coarse graph
        case_bmincut_coarse = FEM.from_couplings(
            'bmincut',
            num_coarse_nodes,
            int(coarse_graph._nnz() // 2),
            coarse_graph,
            node_weights=coarse_node_weights,
        )
        case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q)
        config, result = case_bmincut_coarse.solve()
        
        optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
        best_config = config[optimal_inds[0]]
        coarse_assignment = best_config.argmax(dim=1).cpu().numpy()
        
        # 3. Projection to original graph
        initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
        
        # 4. METIS refinement step
        if not HAS_METIS:
            raise ImportError("pymetis is required for 'coarse_fem_refine_metis' partition method")
        
        adjacency_list = [[] for _ in range(n)]
        indices = J.indices()
        for idx in range(indices.shape[1]):
            r = int(indices[0, idx])
            c = int(indices[1, idx])
            if r != c:  
                adjacency_list[r].append(c)

        # pymetis: call wrapper that passes `part` if supported, otherwise
        # emits a clear warning and calls without initial partition.
        edgecuts, parts = call_pymetis_with_part(q, adjacency_list, part=initial_assignment.tolist())

        # suppressed intermediate prints
        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(parts):
            p[0, i, p_group] = 1.0
            
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        # suppressed intermediate prints

    elif partition_method == 'coarse_fem_refine_kahypar':
        try:
            import kahypar
        except ImportError:
            raise ImportError("kahypar is required for 'coarse_fem_refine_kahypar' partition method")
        start_time = time.time()
        
        from utils import coarsen_graph_by_matching, expand_coarse_labels
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]
        
        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
            J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=500
        )
        num_coarse_nodes = coarse_graph.shape[0]
        
        # Prefer FEM-based Ising/QUBO initial partition on the coarse graph
        # implemented in FEM.initial_partition.fem_initial_partition (k=2).
        # Fall back to the previous FEM-on-coarse solver if anything fails.
        try:
            from FEM.initial_partition import fem_initial_partition

            # Convert sparse coarse_graph to dense numpy adjacency for the QUBO builder
            try:
                coarse_adj_np = coarse_graph.to_dense().cpu().numpy()
            except Exception:
                # If coarse_graph is already dense tensor
                coarse_adj_np = coarse_graph.cpu().numpy()

            c_np = coarse_node_weights.cpu().numpy().reshape(-1)
            coarse_assignment = fem_initial_partition(
                coarse_adj_np,
                None,
                None,
                c_np,
                k=2,
                lambda_penalty=1.0,
                num_trials=num_trials,
                num_steps=num_steps,
                dev=dev,
            )
        except Exception:
            # Fallback: run previous coarse FEM solver
            case_bmincut_coarse = FEM.from_couplings(
                'bmincut', num_coarse_nodes, int(coarse_graph._nnz() // 2), coarse_graph, node_weights=coarse_node_weights
            )
            case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q)
            config, result = case_bmincut_coarse.solve()
            coarse_assignment = config[torch.argwhere(result==result.min()).reshape(-1)[0]].argmax(dim=1).cpu().numpy()
        
        initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
        
        # Kahypar Refinement
        hyperedges = []
        indices = J.indices()
        for idx in range(indices.shape[1]):
            r, c = int(indices[0, idx]), int(indices[1, idx])
            if r < c:
                hyperedges.append([r, c])
                
        num_hyperedges = len(hyperedges)
        hyperedge_indices = []
        hyperedge_indices_ptrs = [0]
        for he in hyperedges:
            hyperedge_indices.extend(he)
            hyperedge_indices_ptrs.append(len(hyperedge_indices))
            
        hypergraph = kahypar.Hypergraph(n, num_hyperedges, hyperedge_indices, hyperedge_indices_ptrs, q, [1]*num_hyperedges, [1]*n)
        for i in range(n):
            hypergraph.setNodePart(i, int(initial_assignment[i]))
            
        context = kahypar.Context()
        try:
            context.loadINIconfiguration("kahypar_config.ini")
        except:
            pass # use defaults
        context.setK(q)
        context.setEpsilon(0.05)
        
        # Improve partition based on the initial block assignments
        kahypar.improvePartition(hypergraph, context)
        part = [hypergraph.blockID(i) for i in range(n)]

        # suppressed intermediate prints
        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(part):
            p[0, i, p_group] = 1.0
            
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        # suppressed intermediate prints

    elif partition_method == 'coarse_fem_refine_kaffpa':
        import kahip
        start_time = time.time()

        from utils import coarsen_graph_by_matching, expand_coarse_labels
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]

        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
            J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=500
        )
        num_coarse_nodes = coarse_graph.shape[0]

        # Use FEM to produce a q-way coarse initial partition so KaFFPa only refines
        from FEM.initial_partition import fem_initial_partition_kway
        try:
            # Convert sparse coarse_graph to dense numpy for the helper
            try:
                coarse_adj_np = coarse_graph.to_dense().cpu().numpy()
            except Exception:
                coarse_adj_np = coarse_graph.cpu().numpy()

            c_np = coarse_node_weights.cpu().numpy().reshape(-1)
            coarse_assignment = fem_initial_partition_kway(
                coarse_adj_np,
                None,
                None,
                c_np,
                k=q,
                lambda_penalty=1.0,
                num_trials=num_trials,
                num_steps=num_steps,
                dev=dev,
            )
        except Exception as e:
            # Let exceptions propagate to surface FEM issues (no silent fallback)
            raise

        initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

        adjacency_list = [[] for _ in range(n)]
        indices = J.indices()
        for idx in range(indices.shape[1]):
            r, c = int(indices[0, idx]), int(indices[1, idx])
            if r != c:  
                adjacency_list[r].append(c)

        xadj = [0]
        adjncy = []
        for r in range(n):
            adjncy.extend(adjacency_list[r])
            xadj.append(len(adjncy))

        vwgt = [1] * n
        adjcwgt = [1] * len(adjncy)

        # Use local simple refinement (KL/FM-like) on top of FEM initial partition
        edgecut, part = simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, part=initial_assignment.tolist(), max_passes=10)

        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(part):
            p[0, i, p_group] = 1.0

        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)

    elif partition_method == 'kaffpa':
        try:
            import kahip
        except ImportError:
            raise ImportError("kahip is required for 'kaffpa' partition method")
        start_time = time.time()
        
        from utils import coarsen_graph_by_matching, expand_coarse_labels
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]
        
        coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
            J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=500
        )
        num_coarse_nodes = coarse_graph.shape[0]
        
        coarse_adj = [[] for _ in range(num_coarse_nodes)]
        c_indices = coarse_graph.indices()
        c_values = coarse_graph.values()
        
        for idx in range(c_indices.shape[1]):
            r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
            if r != c:  
                coarse_adj[r].append((c, int(c_values[idx].item())))
        
        c_xadj = [0]
        c_adjncy = []
        c_adjcwgt = []
        for r in range(num_coarse_nodes):
            for c, w in coarse_adj[r]:
                c_adjncy.append(c)
                c_adjcwgt.append(w)
            c_xadj.append(len(c_adjncy))
            
        c_vwgt = coarse_node_weights.int().cpu().numpy().tolist()
        
        edgecut, coarse_assignment = simple_kaffpa(c_vwgt, c_xadj, c_adjcwgt, c_adjncy, q, epsilon=0.05, max_passes=10)
        coarse_assignment = np.array(coarse_assignment)
        
        initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
        
        adjacency_list = [[] for _ in range(n)]
        indices = J.indices()
        for idx in range(indices.shape[1]):
            r, c = int(indices[0, idx]), int(indices[1, idx])
            if r != c:  
                adjacency_list[r].append(c)
                
        xadj = [0]
        adjncy = []
        for r in range(n):
            adjncy.extend(adjacency_list[r])
            xadj.append(len(adjncy))
            
        vwgt = [1] * n
        adjcwgt = [1] * len(adjncy)
        
        # Use local simple refinement (KL/FM-like) on top of FEM initial partition
        edgecut, part = simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, part=initial_assignment.tolist(), max_passes=10)

        # suppressed intermediate prints
        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(part):
            p[0, i, p_group] = 1.0
            
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        # suppressed intermediate prints

    elif partition_method == 'kahypar':
        try:
            import kahypar
        except ImportError:
            raise ImportError("kahypar is required for 'kahypar' partition method")
        start_time = time.time()
        
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]
        hyperedges = []
        indices = J.indices()
        for idx in range(indices.shape[1]):
            r, c = int(indices[0, idx]), int(indices[1, idx])
            if r < c:
                hyperedges.append([r, c])
                
        num_hyperedges = len(hyperedges)
        hyperedge_indices = []
        hyperedge_indices_ptrs = [0]
        for he in hyperedges:
            hyperedge_indices.extend(he)
            hyperedge_indices_ptrs.append(len(hyperedge_indices))
            
        hypergraph = kahypar.Hypergraph(n, num_hyperedges, hyperedge_indices, hyperedge_indices_ptrs, q, [1]*num_hyperedges, [1]*n)
        context = kahypar.Context()
        try:
            context.loadINIconfiguration("kahypar_config.ini")
        except:
            pass
        context.setK(q)
        context.setEpsilon(0.05)
        
        kahypar.partition(hypergraph, context)
        part = [hypergraph.blockID(i) for i in range(n)]

        # suppressed intermediate prints
        p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
        for i, p_group in enumerate(part):
            p[0, i, p_group] = 1.0
            
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        # suppressed intermediate prints

    else:
        raise ValueError(f"Unknown partition method: {partition_method}")

    if p is None and best_config is not None:
        p = best_config.unsqueeze(0)

    if p is not None:
        # p is of shape (1, n, q) representing permutations
        J = case_bmincut.problem.coupling_matrix
        if not J.is_sparse:
            J = J.to_sparse()
        J = J.coalesce()
        n = J.shape[0]

        final_assignment = p[0].argmax(dim=1).cpu().numpy()
        counts = np.bincount(final_assignment, minlength=q)
        ideal = n / q
        imbalance = float(np.max(np.abs(counts - ideal) / ideal))

        # Evaluate cut value via FEM's infer_bmincut to ensure consistent metric
        from FEM.problem import infer_bmincut
        _, fem_eval_cut = infer_bmincut(J, p)
        try:
            cut_value = float(fem_eval_cut.item())
        except Exception:
            cut_value = float(fem_eval_cut)

        # elapsed time if branch recorded start_time
        try:
            elapsed = time.time() - start_time
        except Exception:
            elapsed = 0.0

        # Print fixed-width table row (trim path from instance name)
        instance_name = os.path.basename(instance)
        col_w = (24, 22, 10, 12, 10)
        row_fmt = f"{{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:>{col_w[2]}.4f}} {{:>{col_w[3]}.1f}} {{:>{col_w[4]}.4f}}"
        print(row_fmt.format(instance_name, partition_method, elapsed, cut_value, imbalance))
