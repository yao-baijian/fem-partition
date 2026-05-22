import numpy as np

from tests.utils import evaluate_kahypar_cut_value, greedy_refine_hypergraph_incremental


def _target_counts(n, q):
    if q <= 0:
        return np.zeros(0, dtype=int)
    base = n // q
    remainder = n % q
    return np.array([base + (1 if i < remainder else 0) for i in range(q)], dtype=int)


def _balance_limits(assignment, max_imbalance, q=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if q is None:
        q = int(assignment.max()) + 1 if assignment.size else 2
    counts = np.bincount(assignment, minlength=q).astype(int)
    ideal = assignment.size / float(q) if q > 0 else 0.0
    max_size = int(np.ceil(ideal * (1.0 + float(max_imbalance)))) if assignment.size else 0
    min_size = int(np.floor(ideal * (1.0 - float(max_imbalance)))) if assignment.size else 0
    return q, counts, min_size, max_size


def _repair_balance_fast(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None):
    """Fast balance repair without cut evaluation."""
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    if assignment.size == 0:
        return assignment

    q, counts, min_size, max_size = _balance_limits(assignment, max_imbalance, q=q)
    node_degree = np.zeros(assignment.size, dtype=float)
    for he in hyperedges:
        for v in he:
            if 0 <= v < assignment.size:
                node_degree[v] += 1.0

    for _ in range(max(1, assignment.size * 2)):
        over = np.where(counts > max_size)[0]
        if len(over) == 0:
            break

        under = np.where(counts < min_size)[0]
        if len(under) == 0:
            under = np.array([int(np.argmin(counts))], dtype=int)

        donor = int(over[np.argmax(counts[over] - max_size)])
        donor_vertices = np.where(assignment == donor)[0]
        if donor_vertices.size == 0:
            break

        rng.shuffle(donor_vertices)
        donor_vertices = donor_vertices[np.argsort(node_degree[donor_vertices], kind='mergesort')]

        moved = False
        for v in donor_vertices:
            for g in under:
                g = int(g)
                if g == donor:
                    continue
                if counts[g] + 1 > max_size:
                    continue
                assignment[v] = g
                counts[donor] -= 1
                counts[g] += 1
                moved = True
                break
            if moved:
                break

        if not moved:
            g = int(np.argmin(counts))
            v = int(donor_vertices[0])
            assignment[v] = g
            counts[donor] -= 1
            counts[g] += 1

    return assignment


def _repair_balance(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None):
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    q, counts, min_size, max_size = _balance_limits(assignment, max_imbalance, q=q)
    targets = _target_counts(len(assignment), q)
    if assignment.size == 0:
        return assignment

    # Iteratively move vertices from surplus groups into deficit groups until exact quotas are met.
    for _ in range(max(1, assignment.size)):
        over = np.where(counts > targets)[0]
        under = np.where(counts < targets)[0]
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
                if counts[g] + 1 > targets[g]:
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


def _partition_summary(assignment, q=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.size == 0:
        return 0, np.zeros(0, dtype=int), 0.0
    if q is None:
        q = int(assignment.max()) + 1
    counts = np.bincount(assignment, minlength=q).astype(int)
    ideal = assignment.size / float(q)
    imb = float(np.max(np.abs(counts - ideal) / ideal)) if ideal > 0 else 0.0
    return q, counts, imb


def refine_with_flow_local_search(assignment, hyperedges, max_passes=3, max_imbalance=0.05, q=None, verbose=False):
    """Flow-inspired local search using a boundary exchange heuristic."""
    if verbose:
        print(f"[refine:flow] start max_passes={max_passes} max_imbalance={max_imbalance}")
    return greedy_refine_hypergraph_incremental(
        assignment,
        hyperedges,
        [1.0] * len(hyperedges),
        q=int(q) if q is not None else (int(np.max(assignment)) + 1 if len(assignment) else 2),
        max_passes=max_passes,
        max_imbalance=max_imbalance,
    )


def refine_with_mcts(assignment, hyperedges, num_rollouts=16, depth=3, seed=None, max_imbalance=0.05, q=None, verbose=False):
    """Monte-Carlo style refinement via randomized move simulations."""
    rng = np.random.default_rng(seed)
    best = np.asarray(assignment, dtype=np.int64).copy()
    base = best.copy()
    best_score = evaluate_kahypar_cut_value(best, hyperedges, [1.0] * len(hyperedges))[0]
    q = int(q) if q is not None else (int(best.max()) + 1 if best.size else 2)
    if best.size:
        node_to_he = [[] for _ in range(best.size)]
        boundary_vertices = set()
        for e_idx, he in enumerate(hyperedges):
            parts = {int(best[v]) for v in he if 0 <= v < best.size}
            if len(parts) > 1:
                for v in he:
                    if 0 <= v < best.size:
                        boundary_vertices.add(int(v))
                    
        boundary_vertices = np.array(sorted(boundary_vertices), dtype=np.int64)
    else:
        boundary_vertices = np.empty((0,), dtype=np.int64)
    if verbose:
        _, _, imb = _partition_summary(best, q=q)
        print(f"[refine:mcts] start rollouts={num_rollouts} depth={depth} cut={best_score} imb={imb:.4f}")
    for _ in range(max(1, int(num_rollouts))):
        cand = best.copy()
        if boundary_vertices.size == 0:
            break
        for _step in range(max(1, int(depth))):
            v = int(boundary_vertices[int(rng.integers(0, boundary_vertices.size))])
            old = int(cand[v])
            new_g = int(rng.integers(0, q - 1))
            if new_g >= old:
                new_g += 1
            if new_g != old:
                cand[v] = new_g
        score = evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0]
        if score < best_score:
            best_score = score
            best = cand
    if _partition_summary(best, q=q)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
    if verbose:
        _, _, imb = _partition_summary(best, q=q)
        print(f"[refine:mcts] done cut={best_score} imb={imb:.4f}")
    return best


def refine_with_evolution(assignment, hyperedges, population_size=8, generations=5, mutation_rate=0.1, seed=None, max_imbalance=0.05, q=None, verbose=False):
    """Small evolutionary search over discrete assignments."""
    rng = np.random.default_rng(seed)
    base = np.asarray(assignment, dtype=np.int64)
    q = int(q) if q is not None else (int(base.max()) + 1 if base.size else 2)
    base_score = evaluate_kahypar_cut_value(base, hyperedges, [1.0] * len(hyperedges))[0]
    _, _, base_imb = _partition_summary(base, q=q)
    low_cut_mode = base_score < 200
    if low_cut_mode:
        mutation_rate = min(float(mutation_rate), 0.01)
        generations = min(int(generations), 3)
    if verbose:
        print(f"[refine:evolution] start pop={population_size} gens={generations} cut={base_score} imb={base_imb:.4f}")

    population = [base.copy() for _ in range(max(1, int(population_size)))]
    mutant_count = max(1, len(population) // 4)
    for idx in range(1, min(len(population), mutant_count + 1)):
        cand = base.copy()
        mask = rng.random(cand.shape[0]) < float(mutation_rate)
        if mask.any():
            cand[mask] = rng.integers(0, q, size=int(mask.sum()))
            if _partition_summary(cand, q=q)[2] > max_imbalance:
                cand = _repair_balance_fast(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
        population[idx] = cand

    for _gen in range(max(1, int(generations))):
        scored = []
        for cand in population:
            score = evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0]
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] > base_score:
            scored = [(base_score, base.copy())] + scored
        elites = [base.copy()]
        elites.extend(cand.copy() for _, cand in scored[: max(1, len(scored) // 3)])
        next_population = elites[:]
        while len(next_population) < len(population):
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = elites[int(rng.integers(0, len(elites)))]
            child = np.where(rng.random(base.shape[0]) < 0.5, p1, p2).copy()
            mut_mask = rng.random(child.shape[0]) < float(mutation_rate)
            if mut_mask.any():
                child[mut_mask] = rng.integers(0, q, size=int(mut_mask.sum()))
                if _partition_summary(child, q=q)[2] > max_imbalance:
                    child = _repair_balance_fast(child, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
            next_population.append(child)
        population = next_population

    scored = [(evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0], cand) for cand in population]
    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]
    if best_score > base_score:
        best = base.copy()
        best_score = base_score
    if _partition_summary(best, q=q)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
    if verbose:
        _, _, imb = _partition_summary(best, q=q)
        print(f"[refine:evolution] done cut={best_score} imb={imb:.4f}")
    return best


def hybrid_refine_partition(
    assignment,
    hyperedges,
    mode_cycle=('mcts', 'evolution', 'flow'),
    rounds=3,
    seed=None,
    max_imbalance=0.05,
    q=None,
    verbose=False,
    mcts_rollouts=16,
    mcts_depth=3,
    evolution_population=8,
    evolution_generations=5,
    evolution_mutation=0.1,
    flow_passes=3,
    skip_exploration_if_good=True,
    good_cut_threshold=200.0,
):
    refined = np.asarray(assignment, dtype=np.int64).copy()
    if q is None:
        q = int(refined.max()) + 1 if refined.size else 2

    if verbose:
        cut, _ = evaluate_kahypar_cut_value(refined, hyperedges, [1.0] * len(hyperedges))
        _, counts, imb = _partition_summary(refined, q=q)
        print(f"[refine:hybrid] start q={q} cut={cut} counts={counts.tolist()} imb={imb:.4f}")

    best = refined.copy()
    best_cut = evaluate_kahypar_cut_value(best, hyperedges, [1.0] * len(hyperedges))[0]
    best_imb = _partition_summary(best, q=q)[2]

    def maybe_repair_and_accept(candidate):
        nonlocal best, best_cut, best_imb
        cand = np.asarray(candidate, dtype=np.int64).copy()
        cand_cut = evaluate_kahypar_cut_value(cand, hyperedges, [1.0] * len(hyperedges))[0]
        _, _, cand_imb = _partition_summary(cand, q=q)

        if cand_imb > max_imbalance:
            if cand_imb > max(0.15, 2.0 * float(max_imbalance)):
                repaired = _repair_balance(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
            else:
                repaired = _repair_balance_fast(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
            repaired_cut = evaluate_kahypar_cut_value(repaired, hyperedges, [1.0] * len(hyperedges))[0]
            _, _, repaired_imb = _partition_summary(repaired, q=q)
            if repaired_cut <= cand_cut:
                cand, cand_cut, cand_imb = repaired, repaired_cut, repaired_imb

        if cand_cut < best_cut or (cand_cut == best_cut and cand_imb <= best_imb):
            best = cand.copy()
            best_cut = float(cand_cut)
            best_imb = float(cand_imb)
            return cand

        return best.copy()

    good_initial = skip_exploration_if_good and best_cut <= float(good_cut_threshold) and best_imb <= float(max_imbalance)
    effective_mode_cycle = ('flow',) if good_initial else tuple(mode_cycle)
    effective_rounds = 1 if good_initial else max(1, int(rounds))
    effective_flow_passes = 1 if good_initial else int(flow_passes)

    for round_idx in range(effective_rounds):
        if verbose:
            print(f"[refine:hybrid] round {round_idx + 1}/{int(effective_rounds)}")

        if 'mcts' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=MCTS")
            candidate = refine_with_mcts(
                refined,
                hyperedges,
                num_rollouts=mcts_rollouts,
                depth=mcts_depth,
                seed=seed,
                max_imbalance=max_imbalance,
                q=q,
                verbose=verbose,
            )
            refined = maybe_repair_and_accept(candidate)

        if 'evolution' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Evolution")
            candidate = refine_with_evolution(
                refined,
                hyperedges,
                population_size=evolution_population,
                generations=evolution_generations,
                mutation_rate=evolution_mutation,
                seed=seed,
                max_imbalance=max_imbalance,
                q=q,
                verbose=verbose,
            )
            refined = maybe_repair_and_accept(candidate)

        if 'flow' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Flow")
            candidate = refine_with_flow_local_search(
                refined,
                hyperedges,
                max_passes=effective_flow_passes,
                max_imbalance=max_imbalance,
                q=q,
                verbose=verbose,
            )
            refined = maybe_repair_and_accept(candidate)

        if _partition_summary(refined, q=q)[2] > max_imbalance:
            refined = _repair_balance_fast(refined, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
        refined = maybe_repair_and_accept(refined)
        if verbose:
            print(f"[refine:hybrid] round_done cut={best_cut} counts={_partition_summary(best, q=q)[1].tolist()} imb={best_imb:.4f}")

    refined = best.copy()
    if _partition_summary(refined, q=q)[2] > max_imbalance:
        refined = _repair_balance_fast(refined, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q)
    refined = maybe_repair_and_accept(refined)
    if verbose:
        cut, _ = evaluate_kahypar_cut_value(refined, hyperedges, [1.0] * len(hyperedges))
        _, counts, imb = _partition_summary(refined, q=q)
        print(f"[refine:hybrid] done cut={cut} counts={counts.tolist()} imb={imb:.4f}")
    return refined
