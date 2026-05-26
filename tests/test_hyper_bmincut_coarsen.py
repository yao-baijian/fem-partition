import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from partition import coarsen_kahypar_like, coarsen_fem_refine_kahypar, evaluate_coarse_cut
from tests.utils import parse_hypergraph_edges


def make_real_hypergraph():
    # load a real hypergraph instance from the benchmark set
    instance = '../partition/full_benchmark_set/powersim.mtx.hgr'
    hyperedges = parse_hypergraph_edges(str(instance))
    num_nodes = max((max(h) for h in hyperedges if h), default=-1) + 1
    return hyperedges, num_nodes


def _run_mode(mode_name, fn, hyperedges, num_nodes, **kwargs):
    res = fn(hyperedges, num_nodes, q=2, coarsen_to=10, verbose=False, **kwargs)
    assert 'coarse_hyperedges' in res
    assert 'initial_assignment' in res
    assert len(res['coarse_groups']) > 0
    cut, imb = evaluate_coarse_cut(res['coarse_hyperedges'], res['initial_assignment'])
    print(f'{mode_name} coarse cut = {cut}, imbalance = {imb}, coarse_nodes = {len(res["coarse_groups"])}')
    return res, cut, imb


def test_compare_coarsen_modes():
    hyperedges, num_nodes = make_real_hypergraph()
    _, kahypar_cut, kahypar_imb = _run_mode('kahypar_like', coarsen_kahypar_like, hyperedges, num_nodes)
    _, fem_hem_cut, fem_hem_imb = _run_mode(
        'coarsen_fem_refine_kahypar[fem_as_hem]',
        coarsen_fem_refine_kahypar,
        hyperedges,
        num_nodes,
        fem_mode='fem_as_hem',
    )
    _, fem_greedy_cut, fem_greedy_imb = _run_mode(
        'coarsen_fem_refine_kahypar[fem_as_greedy_init]',
        coarsen_fem_refine_kahypar,
        hyperedges,
        num_nodes,
        fem_mode='fem_as_greedy_init',
    )

    assert isinstance(kahypar_cut, (int, float)) and kahypar_cut >= 0
    assert isinstance(fem_hem_cut, (int, float)) and fem_hem_cut >= 0
    assert isinstance(fem_greedy_cut, (int, float)) and fem_greedy_cut >= 0
    assert kahypar_imb >= 0 and fem_hem_imb >= 0 and fem_greedy_imb >= 0

    # Comparison is intentionally soft: we only ensure the two coarse stages are both valid
    # and expose their scores for inspection.
    print(
        'comparison: '
        f'kahypar_like={kahypar_cut} vs '
        f'fem_as_hem={fem_hem_cut} vs '
        f'fem_as_greedy_init={fem_greedy_cut}'
    )


if __name__ == '__main__':
    test_compare_coarsen_modes()
    print('smoke ok')