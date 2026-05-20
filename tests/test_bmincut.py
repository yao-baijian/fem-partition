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
# 'direct_fem'                : Original FEM applied directly to normal graph
# 'coarse_fem_refine_metis'   : Multi-level coarsening + FEM coarse opt + METIS fine opt
# 'coarse_fem_refine_kahypar' : Multi-level coarsening + FEM coarse opt + KaHyPar fine opt
# 'coarse_fem_refine_kaffpa'  : Multi-level coarsening + FEM coarse opt + KaFFPa fine opt
# 'metis'                     : PyMetis graph partitioner alone
# 'kahypar'                   : KaHyPar partitioner alone
# 'kaffpa'                    : KaFFPa partitioner alone
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
    _, fem_cut_value = infer_bmincut(case_bmincut.problem.coupling_matrix, best_config.unsqueeze(0))
    fem_cut_value = fem_cut_value.item()
    
    print(f"Direct FEM solve took: {time.time() - start_time:.4f} seconds")
    print(f'{instance}, optimal objective value: {result.min()}, actual edge cut: {fem_cut_value}')

elif partition_method == 'metis':
    if not HAS_METIS:
        raise ImportError("pymetis is required for 'metis' partition method")
    print("====== Running METIS ======")
    
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
    print(f"METIS solve took: {time.time() - start_time:.4f} seconds")
    
    # evaluate METIS assignment cut with FEM traditional bmincut cut
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(parts):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    fem_eval_cut = fem_eval_cut.item()
    
    print(f'{instance}, METIS edgecuts reported: {edgecuts}, Evaluated actual edge cut: {fem_eval_cut}')

elif partition_method == 'coarse_fem_refine_metis':
    print("====== Running Coarse FEM + METIS Refinement ======")
    start_time = time.time()
    
    from utils import coarsen_graph_by_leaf_folding, expand_coarse_labels
    J = case_bmincut.problem.coupling_matrix
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    # 1. Multi-level coarsening
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_leaf_folding(
        J,
        node_weights=torch.ones(n, dtype=torch.float32),
        max_degree=500,
        min_nodes=5000,
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

    # pymetis (passing initial part if supported, otherwise falling back)
    try:
        edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list, part=initial_assignment.tolist())
    except TypeError:
        # Fallback if pymetis doesn't accept 'part' kwarg
        edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list)

    print(f"Solve took: {time.time() - start_time:.4f} seconds")
    
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(parts):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    print(f'{instance}, Evaluated actual edge cut: {fem_eval_cut.item()}')

elif partition_method == 'coarse_fem_refine_kahypar':
    try:
        import kahypar
    except ImportError:
        raise ImportError("kahypar is required for 'coarse_fem_refine_kahypar' partition method")
    print("====== Running Coarse FEM + KaHyPar Refinement ======")
    start_time = time.time()
    
    from utils import coarsen_graph_by_leaf_folding, expand_coarse_labels
    J = case_bmincut.problem.coupling_matrix
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_leaf_folding(
        J, node_weights=torch.ones(n, dtype=torch.float32), max_degree=500, min_nodes=5000
    )
    num_coarse_nodes = coarse_graph.shape[0]
    
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

    print(f"Solve took: {time.time() - start_time:.4f} seconds")
    
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(part):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    print(f'{instance}, Evaluated actual edge cut: {fem_eval_cut.item()}')

elif partition_method == 'coarse_fem_refine_kaffpa':
    try:
        import kahip
    except ImportError:
        raise ImportError("kahip is required for 'coarse_fem_refine_kaffpa' partition method")
    print("====== Running Coarse FEM + KaFFPa Refinement ======")
    start_time = time.time()
    
    from utils import coarsen_graph_by_leaf_folding, expand_coarse_labels
    J = case_bmincut.problem.coupling_matrix
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_leaf_folding(
        J, node_weights=torch.ones(n, dtype=torch.float32), max_degree=500, min_nodes=5000
    )
    num_coarse_nodes = coarse_graph.shape[0]
    
    case_bmincut_coarse = FEM.from_couplings(
        'bmincut', num_coarse_nodes, int(coarse_graph._nnz() // 2), coarse_graph, node_weights=coarse_node_weights
    )
    case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q)
    config, result = case_bmincut_coarse.solve()
    coarse_assignment = config[torch.argwhere(result==result.min()).reshape(-1)[0]].argmax(dim=1).cpu().numpy()
    
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
    
    # Normally KaHiP might not natively expose initialize node partition via simple python kaffpa wrapper.
    # Attempting an assignment logic if `part` or `initial_part` exists in bindings, else just run from scratch
    # kaffpa parameters: (vwgt, xadj, adjcwgt, adjncy, nparts, imbalance, suppress_output, seed, mode)
    try:
        edgecut, part = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, q, 0.05, False, 0, kahip.FASTSOCIAL, part=initial_assignment.tolist())
    except TypeError:
        # Fallback to standard kaffpa if initial partition passing is not structurally supported in wrapper
        edgecut, part = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, q, 0.05, False, 0, kahip.FASTSOCIAL)

    print(f"Solve took: {time.time() - start_time:.4f} seconds")
    
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(part):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    print(f'{instance}, Evaluated actual edge cut: {fem_eval_cut.item()}')

elif partition_method == 'kahypar':
    try:
        import kahypar
    except ImportError:
        raise ImportError("kahypar is required for 'kahypar' partition method")
    print("====== Running KaHyPar Alone ======")
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

    print(f"KaHyPar solve took: {time.time() - start_time:.4f} seconds")
    
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(part):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    print(f'{instance}, Evaluated actual edge cut: {fem_eval_cut.item()}')

elif partition_method == 'kaffpa':
    try:
        import kahip
    except ImportError:
        raise ImportError("kahip is required for 'kaffpa' partition method")
    print("====== Running KaFFPa Alone ======")
    start_time = time.time()
    
    J = case_bmincut.problem.coupling_matrix
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
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
    
    edgecut, part = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, q, 0.05, False, 0, kahip.FASTSOCIAL)

    print(f"KaFFPa solve took: {time.time() - start_time:.4f} seconds")
    
    p = torch.zeros((1, n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(part):
        p[0, i, p_group] = 1.0
        
    from FEM.problem import infer_bmincut
    _, fem_eval_cut = infer_bmincut(J, p)
    print(f'{instance}, Evaluated actual edge cut: {fem_eval_cut.item()}')

else:
    raise ValueError(f"Unknown partition method: {partition_method}")
