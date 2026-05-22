import numpy as np

from tests.utils import evaluate_kahypar_cut_value, greedy_refine_hypergraph_incremental


def _balance_limits(assignment, max_imbalance, q=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if q is None:
        q = int(assignment.max()) + 1 if assignment.size else 2
    counts = np.bincount(assignment, minlength=q).astype(int)
    ideal = assignment.size / float(q) if q > 0 else 0.0
    max_size = int(np.ceil(ideal * (1.0 + float(max_imbalance)))) if assignment.size else 0
    min_size = int(np.floor(ideal * (1.0 - float(max_imbalance)))) if assignment.size else 0
    return q, counts, min_size, max_size


def _repair_balance(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None):
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    q, counts, min_size, max_size = _balance_limits(assignment, max_imbalance, q=q)
    if assignment.size == 0:
        return assignment

    # Iteratively move vertices from oversized groups into undersized groups.
    for _ in range(max(1, assignment.size)):
        over = np.where(counts > max_size)[0]
        under = np.where(counts < min_size)[0]
        if len(over) == 0 or len(under) == 0:
            break

        moved = False
        candidates = np.where(np.isin(assignment, over))[0]
        rng.shuffle(candidates)
        for v in candidates:
            old = int(assignment[v])
            best_g = None
            best_score = None
            for g in under:
                if counts[g] + 1 > max_size:
                    continue
                trial = assignment.copy()
                trial[v] = int(g)
                score = evaluate_kahypar_cut_value(trial, hyperedges, [1.0] * len(hyperedges))[0]
                if best_score is None or score < best_score:
                    best_score = score
                    best_g = int(g)
            if best_g is not None:
                assignment[v] = best_g
                counts[old] -= 1
                counts[best_g] += 1
                moved = True
                break
        if not moved:
            break

    return assignment


def refine_with_flow_local_search(assignment, hyperedges, max_passes=3, max_imbalance=0.05, q=None):
    """Flow-inspired local search using a boundary exchange heuristic."""
    return greedy_refine_hypergraph_incremental(
        assignment,
        hyperedges,
        [1.0] * len(hyperedges),
        q=int(q) if q is not None else (int(np.max(assignment)) + 1 if len(assignment) else 2),
        max_passes=max_passes,
        max_imbalance=max_imbalance,
    )


def refine_with_mcts(assignment, hyperedges, num_rollouts=16, depth=3, seed=None, max_imbalance=0.05, q=None):
    """Monte-Carlo style refinement via randomized move simulations."""
    rng = np.random.default_rng(seed)
    best = np.asarray(assignment, dtype=np.int64).copy()
    best_score = evaluate_kahypar_cut_value(best, hyperedges, [1.0] * len(hyperedges))[0]
    q = int(q) if q is not None else (int(best.max()) + 1 if best.size else 2)
    _, counts, _, max_size = _balance_limits(best, max_imbalance, q=q)
    for _ in range(max(1, int(num_rollouts))):
        cand = best.copy()
        for _step in range(max(1, int(depth))):
            v = int(rng.integers(0, len(cand)))
            old = int(cand[v])
            new_g = int(rng.integers(0, q))
            if new_g != old and counts[new_g] + 1 <= max_size:
                cand[v] = new_g
        score = evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0]
        if score < best_score:
            best_score = score
            best = cand
    return _repair_balance(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)


def refine_with_evolution(assignment, hyperedges, population_size=8, generations=5, mutation_rate=0.1, seed=None, max_imbalance=0.05, q=None):
    """Small evolutionary search over discrete assignments."""
    rng = np.random.default_rng(seed)
    base = np.asarray(assignment, dtype=np.int64)
    q = int(q) if q is not None else (int(base.max()) + 1 if base.size else 2)
    _, counts, _, max_size = _balance_limits(base, max_imbalance, q=q)
    population = [base.copy()]
    for _ in range(max(0, int(population_size) - 1)):
        cand = base.copy()
        mask = rng.random(cand.shape[0]) < float(mutation_rate)
        new_values = rng.integers(0, q, size=int(mask.sum()))
        if mask.any():
            cand[mask] = new_values
            cand = _repair_balance(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
        population.append(cand)

    for _gen in range(max(1, int(generations))):
        scored = []
        for cand in population:
            score = evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0]
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0])
        elites = [cand.copy() for _, cand in scored[: max(1, len(scored) // 3)]]
        next_population = elites[:]
        while len(next_population) < len(population):
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = elites[int(rng.integers(0, len(elites)))]
            child = np.where(rng.random(base.shape[0]) < 0.5, p1, p2).copy()
            mut_mask = rng.random(child.shape[0]) < float(mutation_rate)
            if mut_mask.any():
                child[mut_mask] = rng.integers(0, q, size=int(mut_mask.sum()))
                child = _repair_balance(child, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
            next_population.append(child)
        population = next_population

    scored = [(evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0], cand) for cand in population]
    scored.sort(key=lambda x: x[0])
    return _repair_balance(scored[0][1], hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)


def hybrid_refine_partition(assignment, hyperedges, mode_cycle=('flow', 'mcts', 'evolution'), rounds=3, seed=None, max_imbalance=0.05, q=None):
    refined = np.asarray(assignment, dtype=np.int64).copy()
    for i in range(max(1, int(rounds))):
        mode = mode_cycle[i % len(mode_cycle)]
        if mode == 'flow':
            refined = refine_with_flow_local_search(refined, hyperedges, max_imbalance=max_imbalance, q=q)
        elif mode == 'mcts':
            refined = refine_with_mcts(refined, hyperedges, seed=seed, max_imbalance=max_imbalance, q=q)
        elif mode == 'evolution':
            refined = refine_with_evolution(refined, hyperedges, seed=seed, max_imbalance=max_imbalance, q=q)
    return _repair_balance(refined, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
