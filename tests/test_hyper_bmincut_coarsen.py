import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from partition import coarsen_kahypar_like, coarsen_fem_refine_kahypar, evaluate_coarse_cut


def make_tiny_hypergraph():
    # small synthetic hypergraph for fast unit tests
    hyperedges = [[0, 1, 2], [2, 3], [1, 3]]
    num_nodes = 4
    return hyperedges, num_nodes


def _run_mode(mode_name, fn, hyperedges, num_nodes):
    res = fn(hyperedges, num_nodes, q=2, coarsen_to=10, verbose=False)
    assert 'coarse_hyperedges' in res
    assert 'initial_assignment' in res
    cut, imb = evaluate_coarse_cut(res['coarse_hyperedges'], res['initial_assignment'])
    print(f'{mode_name} coarse cut = {cut}, imbalance = {imb}, coarse_nodes = {len(res["coarse_groups"])}')
    return res, cut, imb


def test_compare_coarsen_modes():
    hyperedges, num_nodes = make_tiny_hypergraph()
    _, kahypar_cut, kahypar_imb = _run_mode('kahypar_like', coarsen_kahypar_like, hyperedges, num_nodes)
    _, fem_cut, fem_imb = _run_mode('coarsen_fem_refine_kahypar', coarsen_fem_refine_kahypar, hyperedges, num_nodes)

    assert isinstance(kahypar_cut, (int, float)) and kahypar_cut >= 0
    assert isinstance(fem_cut, (int, float)) and fem_cut >= 0
    assert kahypar_imb >= 0 and fem_imb >= 0

    # Comparison is intentionally soft: we only ensure the two coarse stages are both valid
    # and expose their scores for inspection.
    print(f'comparison: kahypar_like={kahypar_cut} vs coarsen_fem_refine_kahypar={fem_cut}')


if __name__ == '__main__':
    test_compare_coarsen_modes()
    print('smoke ok')