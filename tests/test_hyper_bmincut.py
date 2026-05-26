import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / 'tests'
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from FEM import FEM
import torch
import time
import numpy as np
import warnings

from partition import coarsen_kahypar_like as shared_kahypar_like_coarsen
from partition import coarsen_fem_refine_kahypar as shared_fem_matching_coarsen
from partition.helper import build_coarse_hyperedges, make_q4_pubo_object
from partition.refine import hybrid_refine_partition
from utils import (
    build_clique_expanded_graph,
    coarsen_graph_by_matching,
    evaluate_kahypar_cut_value,
    expand_coarse_labels,
    greedy_refine_hypergraph_incremental,
    parse_hypergraph_edges,
    PUBOObjective,
)

try:
    import kahypar  # type: ignore[import-not-found]

    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False
    warnings.warn("KaHyPar is not installed. Will fallback to FEM where applicable.")

try:
    import pymetis  # type: ignore[import-not-found]

    HAS_METIS = True
except ImportError:
    HAS_METIS = False


num_trials = 1
num_steps = 20
dev = 'cpu'
instance = '../partition/full_benchmark_set/powersim.mtx.hgr'

# ==========================================
# Select the partition method(s) to run:
# 'direct_fem'             : Original FEM applied directly to the clique-expanded hypergraph
# 'coarsen_fem_refine_kahypar' : QUBO-based matching coarsening (FEM) + KaHyPar on coarse hypergraph
# 'coarsen_kahypar_refine' : Multi-level coarsening + KaHyPar initial guess + Greedy refinement
# 'kahyper_like'           : Self-implemented KaHyPar-like coarsening + greedy coarse solve + greedy refinement
# 'kahyper_like_no_lsh'    : Same as above but with LSH disabled (pure Heavy Edge Matching)
# 'pubo_direct'            : Full PUBO-based objective directly on hypergraph (Auto Grad + Opt)
# 'pubo_coarsen'           : Coarsening framework + PUBO on the compressed hyperedges
# 'pubo_q4_explicit'       : Coarsening + explicit formulation via expected_hyperbmincut_explicit
# 'pubo_implicit'          : Coarsening + approximate formulation via expected_hyperbmincut
# ==========================================
partition_methods = [
    # 'direct_fem',
    'coarsen_fem_refine_kahypar',
    'coarsen_kahypar_refine',
    'kahyper_like',
    'kahyper_like_no_lsh',
    # 'pubo_direct',
    # 'pubo_coarsen',
    # 'pubo_q4_explicit',
    # 'pubo_implicit',
]

verbose = False

# Whether to use LSH pre-clustering before Heavy Edge Matching.
# When False (default for kahyper_like_no_lsh), HEM runs directly on the original hypergraph.
# When True, MinHash/LSH is used to bucket similar vertices first.
use_lsh = False


def _log(message, enabled=False):
    if enabled:
        print(message)


def _print_results_table(rows):
    col_w = (30, 28, 10, 12, 10)
    header_fmt = f"{{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}}"
    sep = ' '.join(['-' * w for w in col_w])
    print(header_fmt.format('instance', 'method', 'time(s)', 'cut', 'imbalance'))
    print(sep)
    for row in rows:
        print(
            header_fmt.format(
                row['instance'],
                row['method'],
                f"{row['time_s']:.4f}",
                f"{row['cut']:.4f}",
                f"{row['imbalance']:.6f}",
            )
        )


def run_kahypar_like_multilevel(
    clique_graph_local,
    hyperedges_local,
    num_nodes_local,
    q_local,
    coarsen_to=500,
    verbose=False,
    use_lsh=False,
):
    """Run KaHyPar-like multilevel coarsening with optional LSH preprocessing."""
    stage_t0 = time.time()
    res = shared_kahypar_like_coarsen(
        hyperedges_local,
        num_nodes_local,
        q=q_local,
        coarsen_to=max(10, int(coarsen_to)),
        verbose=verbose,
        use_lsh=use_lsh,
    )
    _log(
        (
            f"[kahyper_like] shared_kahypar_like_coarsen: "
            f"n={num_nodes_local} -> {len(res['coarse_groups'])}, "
            f"nnz={int(res['coarse_graph']._nnz()) if res['coarse_graph'].is_sparse else 0}, "
            f"time={time.time() - stage_t0:.4f}s"
        ),
        verbose,
    )

    return (
        res['coarse_graph'],
        res['coarse_node_weights'],
        res['coarse_groups'],
        res['original_to_coarse'],
        res['initial_assignment'],
    )


def _compute_summary(final_assignment, hyperedges, q_ways):
    fem_cut_value, _ = evaluate_kahypar_cut_value(final_assignment, hyperedges, [1.0] * len(hyperedges))
    counts = np.bincount(final_assignment, minlength=q_ways)
    ideal = len(final_assignment) / q_ways
    max_imbalance = float(np.max(np.abs(counts - ideal) / ideal)) if ideal > 0 else 0.0
    return float(fem_cut_value), max_imbalance


def run_partition_method(partition_method, hyperedges, clique_graph, num_nodes, q_ways, verbose=False):
    requested_method = partition_method
    start_time = time.time()
    log = lambda msg: _log(msg, verbose)

    log(f"Loading {instance}...")
    log(f"====== Running {partition_method} ======")

    if partition_method == 'direct_fem':
        log("====== Running Direct FEM ======")
        graph_for_fem = clique_graph
        node_weights_for_fem = torch.ones(num_nodes, dtype=torch.float32)

        log("Setting up FEM solver...")
        case_bmincut = FEM.from_couplings(
            'bmincut',
            graph_for_fem.shape[0],
            int(clique_graph._nnz() // 2),
            graph_for_fem,
            node_weights=node_weights_for_fem,
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=True)

        log("Running FEM optimize...")
        config, result = case_bmincut.solve()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        best_config = config[optimal_inds[0]]
        final_assignment = best_config.argmax(dim=1).cpu().numpy()

    elif partition_method == 'pubo_direct':
        log("====== Running Direct PUBO FEM ======")
        pubo_obj = PUBOObjective(
            hyperedges,
            [1.0] * len(hyperedges),
            q=q_ways,
            num_nodes=num_nodes,
            node_weights=torch.ones(num_nodes, dtype=torch.float32),
            imbalance_weight=5.0,
            obj_type='cut_net',
            max_degree=5,
        )

        dummy_matrix = torch.zeros((num_nodes, num_nodes))
        case_bmincut = FEM()
        case_bmincut.set_up_problem(
            num_nodes,
            0,
            'customize',
            dummy_matrix,
            q=q_ways,
            customize_expected_func=pubo_obj.expectation,
            customize_infer_func=pubo_obj.inference,
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)

        log("Running PUBO FEM optimize...")
        config, result = case_bmincut.solve()
        best_config = config[2] if len(config) > 2 else config[0]
        final_assignment = best_config.argmax(dim=1).cpu().numpy()

    elif partition_method in ['coarsen_fem_refine_kahypar', 'coarsen_kahypar_refine', 'kahyper_like', 'kahyper_like_no_lsh', 'pubo_coarsen', 'pubo_q4_explicit', 'pubo_implicit']:
        log(f"====== Running {partition_method} ======")
        if partition_method in ['coarsen_kahypar_refine', 'kahyper_like', 'kahyper_like_no_lsh']:
            coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, initial_assignment = run_kahypar_like_multilevel(
                clique_graph,
                hyperedges,
                num_nodes,
                q_ways,
                coarsen_to=30,
                verbose=verbose,
                use_lsh=(partition_method == 'kahyper_like'),
            )
            num_coarse_nodes = coarse_graph.shape[0]
            log(f"KaHyPar-like coarse partitioning took: {time.time() - start_time:.4f} seconds")
        elif partition_method == 'coarsen_fem_refine_kahypar':
            shared_res = shared_fem_matching_coarsen(
                hyperedges,
                num_nodes,
                q=q_ways,
                coarsen_to=500,
                num_trials=num_trials,
                num_steps=num_steps,
                dev=dev,
                verbose=verbose,
            )
            coarse_graph = shared_res['coarse_graph']
            coarse_node_weights = shared_res['coarse_node_weights']
            coarse_groups = shared_res['coarse_groups']
            original_to_coarse = shared_res['original_to_coarse']
            initial_assignment = shared_res['initial_assignment']
            num_coarse_nodes = coarse_graph.shape[0]
            log(f"Shared FEM matching coarsening took: {time.time() - start_time:.4f} seconds")
        else:
            coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
                clique_graph,
                node_weights=torch.ones(num_nodes, dtype=torch.float32),
                coarsen_to=500,
            )
            num_coarse_nodes = coarse_graph.shape[0]

        use_kahypar_refine = False
        if partition_method == 'coarsen_kahypar_refine' and HAS_KAHYPAR:
            log("Using KaHyPar for refinement when available (will use FEM for coarse initial assignment).")
            use_kahypar_refine = True
        elif partition_method == 'coarsen_kahypar_refine' and not HAS_KAHYPAR:
            log("KaHyPar is requested but not installed. Falling back to FEM on the coarsened graph (coarsen_fem_refine).")
            partition_method = 'coarsen_fem_refine'

        if partition_method == 'pubo_coarsen':
            log("Using PUBO as the primary solver on the coarsened graph...")
            coarse_hyperedges = build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)

            pubo_obj = PUBOObjective(
                coarse_hyperedges,
                [1.0] * len(coarse_hyperedges),
                q=q_ways,
                num_nodes=num_coarse_nodes,
                node_weights=coarse_node_weights,
                imbalance_weight=5.0,
                obj_type='cut_net',
                max_degree=5,
            )

            dummy_matrix = torch.zeros((num_coarse_nodes, num_coarse_nodes))
            case_bmincut = FEM()
            case_bmincut.set_up_problem(
                num_coarse_nodes,
                0,
                'customize',
                dummy_matrix,
                q=q_ways,
                customize_expected_func=pubo_obj.expectation,
                customize_infer_func=pubo_obj.inference,
            )
            case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
            config, result = case_bmincut.solve()
            best_config = config[0]
            initial_assignment = best_config.argmax(dim=1).cpu().numpy()
            log(f"Coarse PUBO partitioning took: {time.time() - start_time:.4f} seconds")

        if partition_method == 'pubo_q4_explicit':
            log("Using Explicit q=4 PUBO on the coarsened graph...")
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
                num_coarse_nodes,
                0,
                'customize',
                dummy_matrix,
                q=q_ways,
                customize_expected_func=pubo_obj.expectation,
                customize_infer_func=pubo_obj.inference,
            )
            case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
            config, result = case_bmincut.solve()
            best_config = config[0]
            initial_assignment = best_config.argmax(dim=1).cpu().numpy()
            log(f"Explicit q=4 PUBO partitioning took: {time.time() - start_time:.4f} seconds")

        if partition_method == 'pubo_implicit':
            log("Using implicit q=4 PUBO on the coarsened graph...")
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
                num_coarse_nodes,
                0,
                'customize',
                dummy_matrix,
                q=q_ways,
                customize_expected_func=pubo_obj.expectation,
                customize_infer_func=pubo_obj.inference,
            )
            case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
            config, result = case_bmincut.solve()
            best_config = config[0]
            initial_assignment = best_config.argmax(dim=1).cpu().numpy()
            log(f"Implicit q=4 PUBO partitioning took: {time.time() - start_time:.4f} seconds")

        log("Step 3: Uncoarsening (Projection) back to original hypergraph...")
        step3_t0 = time.time()
        group_assignment = expand_coarse_labels(coarse_groups, initial_assignment, num_nodes)
        log(f"Step 3: expand_coarse_labels finished in {time.time() - step3_t0:.4f}s")

        if requested_method in ('coarsen_kahypar_refine', 'coarsen_fem_refine_kahypar') and HAS_KAHYPAR and use_kahypar_refine:
            log("Step 3: Running KaHyPar refinement on the original hypergraph...")
            hyperedges_indices = []
            hyperedges_ptrs = [0]
            for he in hyperedges:
                hyperedges_indices.extend(he)
                hyperedges_ptrs.append(len(hyperedges_indices))

            hg = kahypar.Hypergraph(num_nodes, len(hyperedges), hyperedges_indices, hyperedges_ptrs, q_ways, [1] * len(hyperedges), [1] * num_nodes)
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
            final_assignment = np.array([hg.blockID(i) for i in range(num_nodes)], dtype=np.int64)
        else:
            log("Step 3: Running Hybrid Refinement (Flow + MCTS + Evolution)...")
            step3_refine_t0 = time.time()
            if requested_method in ('kahyper_like', 'kahyper_like_no_lsh'):
                final_assignment = hybrid_refine_partition(
                    group_assignment,
                    hyperedges,
                    mode_cycle=('flow', 'mcts', 'evolution'),
                    rounds=3,
                    q=q_ways,
                    verbose=verbose,
                )
                log(f"Step 3: hybrid_refine_partition finished in {time.time() - step3_refine_t0:.4f}s")
            elif requested_method == 'coarsen_fem_refine_kahypar':
                final_assignment = hybrid_refine_partition(
                    group_assignment,
                    hyperedges,
                    mode_cycle=('flow',),
                    rounds=1,
                    q=q_ways,
                    verbose=verbose,
                    flow_passes=2,
                    skip_exploration_if_good=True,
                    good_cut_threshold=200.0,
                )
                log(f"Step 3: flow-only refinement finished in {time.time() - step3_refine_t0:.4f}s")
            else:
                final_assignment = greedy_refine_hypergraph_incremental(
                    group_assignment,
                    hyperedges,
                    [1.0] * len(hyperedges),
                    q=q_ways,
                    max_passes=5,
                    max_imbalance=0.05,
                )
                log(f"Step 3: greedy_refine_hypergraph_incremental finished in {time.time() - step3_refine_t0:.4f}s")
    else:
        raise ValueError(f"Unknown partition method: {requested_method}")

    cut_value, max_imbalance = _compute_summary(final_assignment, hyperedges, q_ways)
    elapsed = time.time() - start_time
    return {
        'instance': instance,
        'method': requested_method,
        'time_s': elapsed,
        'cut': cut_value,
        'imbalance': max_imbalance,
        'assignment': final_assignment,
    }


hyperedges = parse_hypergraph_edges(instance)
num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1
clique_graph = build_clique_expanded_graph(hyperedges, num_nodes=num_nodes, normalize_weight=True)

q_ways = 4

rows = []
for partition_method in partition_methods:
    rows.append(run_partition_method(partition_method, hyperedges, clique_graph, num_nodes, q_ways, verbose=verbose))

_print_results_table(rows)
