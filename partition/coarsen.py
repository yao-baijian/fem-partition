import heapq
import math

import numpy as np
import torch

from tests.utils import build_clique_expanded_graph, evaluate_kahypar_cut_value, greedy_initial_hypergraph_partition


def _build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes):
    coarse_hyperedges = []
    for he in hyperedges:
        coarse_he = []
        seen = set()
        for v in he:
            if v < num_nodes:
                c = int(original_to_coarse[v])
                if c not in seen:
                    coarse_he.append(c)
                    seen.add(c)
        if len(coarse_he) > 1:
            coarse_hyperedges.append(coarse_he)
    return coarse_hyperedges


def _evaluate_pair_rating(u, v, alive, vertex_to_edges, edge_vertices, edge_weights):
    if not alive.get(u, False) or not alive.get(v, False) or u == v:
        return 0.0
    common = vertex_to_edges.get(u, set()) & vertex_to_edges.get(v, set())
    rating = 0.0
    for eid in common:
        verts = edge_vertices.get(eid)
        if not verts:
            continue
        size = len(verts)
        if size > 1:
            rating += float(edge_weights[eid]) / float(size - 1)
    return rating


def _push_pair(heap, pair_rating, u, v, rating):
    if u == v or rating <= 0.0:
        return
    a, b = (u, v) if u < v else (v, u)
    pair_rating[(a, b)] = float(rating)
    heapq.heappush(heap, (-float(rating), a, b))


def _vertex_feature_matrix(hyperedges, num_nodes):
    features = np.zeros((num_nodes, 4), dtype=np.float32)
    for he in hyperedges:
        size = float(max(1, len(he)))
        edge_weight = 1.0
        for v in he:
            if 0 <= v < num_nodes:
                features[v, 0] += 1.0
                features[v, 1] += size
                features[v, 2] += edge_weight
                features[v, 3] += 1.0 / size
    row_norm = np.linalg.norm(features, axis=1, keepdims=True)
    row_norm[row_norm == 0.0] = 1.0
    return features / row_norm


def _vertex_incident_edge_sets(hyperedges, num_nodes):
    incident = [set() for _ in range(num_nodes)]
    for eid, he in enumerate(hyperedges):
        for v in he:
            if 0 <= v < num_nodes:
                incident[v].add(eid)
    return incident


def _minhash_signatures(incident_edge_sets, num_hashes=128, seed=None):
    rng = np.random.default_rng(seed)
    num_vertices = len(incident_edge_sets)
    if num_vertices == 0:
        return np.empty((0, 0), dtype=np.uint64)

    num_edges = 1 + max((max(s) for s in incident_edge_sets if s), default=-1)
    if num_edges <= 0:
        return np.zeros((num_vertices, max(1, int(num_hashes))), dtype=np.uint64)

    num_hashes = max(1, int(num_hashes))
    # Use a universal hash family over edge ids.
    prime = np.uint64(4294967311)
    a = rng.integers(1, int(prime - 1), size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, int(prime - 1), size=num_hashes, dtype=np.uint64)

    edge_ids = np.arange(num_edges, dtype=np.uint64)
    hashes = ((a[:, None] * edge_ids[None, :] + b[:, None]) % prime).astype(np.uint64)

    signatures = np.full((num_vertices, num_hashes), np.uint64(prime - 1), dtype=np.uint64)
    for v, edges in enumerate(incident_edge_sets):
        if not edges:
            continue
        edge_idx = np.fromiter(edges, dtype=np.uint64)
        signatures[v, :] = hashes[:, edge_idx].min(axis=1)
    return signatures


def _jaccard_similarity(edge_set_a, edge_set_b):
    if not edge_set_a and not edge_set_b:
        return 1.0
    union = edge_set_a | edge_set_b
    if not union:
        return 0.0
    inter = edge_set_a & edge_set_b
    return float(len(inter)) / float(len(union))


def _lsh_groups_from_incident_sets(
    incident_edge_sets,
    num_hashes=128,
    num_bands=32,
    rows_per_band=4,
    target_buckets=200,
    threshold=0.5,
    min_threshold=0.1,
    seed=None,
    verbose=False,
):
    """Build merge groups via standard MinHash-LSH banding plus exact Jaccard filtering."""
    num_vertices = len(incident_edge_sets)
    if num_vertices == 0:
        return []

    num_hashes = max(1, int(num_hashes))
    num_bands = max(1, int(num_bands))
    rows_per_band = max(1, int(rows_per_band))
    target_buckets = max(1, int(target_buckets))
    min_threshold = float(min_threshold)

    # Keep the MinHash layout consistent with requested banding.
    required_hashes = num_bands * rows_per_band
    if num_hashes < required_hashes:
        num_hashes = required_hashes

    signatures = _minhash_signatures(incident_edge_sets, num_hashes=num_hashes, seed=seed)
    num_bucket_slots = max(
        128,
        min(num_vertices, int(np.ceil(float(num_vertices) / max(1.0, float(target_buckets) / float(num_bands))))),
    )

    def _compute_groups(thr):
        # 1) Band buckets via second-level hashing.
        buckets = {}
        max_band_rows = signatures.shape[1]
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            if end > max_band_rows:
                break
            for v in range(num_vertices):
                band_slice = tuple(int(x) for x in signatures[v, start:end])
                band_hash = hash((band_idx, band_slice)) % num_bucket_slots
                key = (band_idx, band_hash)
                buckets.setdefault(key, []).append(v)

        # 2) Candidate set from all intra-bucket pairs.
        candidates = set()
        for verts in buckets.values():
            if len(verts) <= 1:
                continue
            unique_verts = sorted(set(verts))
            for i in range(len(unique_verts)):
                u = unique_verts[i]
                for j in range(i + 1, len(unique_verts)):
                    v = unique_verts[j]
                    candidates.add((u, v))

        # 3) Union-find using exact Jaccard thresholding.
        parent = np.arange(num_vertices, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        hits = 0
        checks = 0
        for u, v in candidates:
            checks += 1
            sim = _jaccard_similarity(incident_edge_sets[u], incident_edge_sets[v])
            if sim >= thr:
                hits += 1
                union(u, v)

        comp = {}
        for v in range(num_vertices):
            root = find(v)
            comp.setdefault(root, []).append(v)

        groups_local = list(comp.values())
        return groups_local, len(candidates), checks, hits

    cur_threshold = float(threshold)
    groups, candidate_pairs, checks, hits = _compute_groups(cur_threshold)

    # 4) Optional threshold relaxation toward target bucket count.
    while len(groups) > target_buckets and cur_threshold > min_threshold:
        cur_threshold = max(min_threshold, cur_threshold * 0.8)
        groups, candidate_pairs, checks, hits = _compute_groups(cur_threshold)

    if verbose and checks > 0:
        hit_ratio = float(hits) / float(checks)
        print(
            f"[kahypar_like] LSH/Jaccard threshold sanity: threshold={cur_threshold:.3f}, "
            f"hit_ratio={hit_ratio:.3f}, candidates={candidate_pairs}, buckets={len(groups)}, target={target_buckets}"
        )

    return groups


def _lsh_bucketize_vertices(
    hyperedges,
    num_nodes,
    target_buckets=None,
    num_planes=4,
    num_tables=32,
    seed=None,
    jaccard_threshold=0.1,
    num_hashes=128,
    verbose=False,
):
    """Pre-coarsen vertices using MinHash/LSH over incident hyperedge sets."""
    if num_nodes == 0:
        return np.arange(0, dtype=np.int64), []

    incident_edge_sets = _vertex_incident_edge_sets(hyperedges, num_nodes)
    groups = _lsh_groups_from_incident_sets(
        incident_edge_sets,
        num_hashes=num_hashes,
        num_bands=max(1, int(num_tables)),
        rows_per_band=max(1, int(num_planes)),
        target_buckets=max(1, int(target_buckets)) if target_buckets is not None else max(1, int(num_nodes // 4)),
        threshold=float(jaccard_threshold),
        min_threshold=0.02,
        seed=seed,
        verbose=verbose,
    )

    original_to_bucket = np.empty(num_nodes, dtype=np.int64)
    for idx, verts in enumerate(groups):
        for v in verts:
            original_to_bucket[v] = idx

    return original_to_bucket, groups


def _rebuild_hyperedges_from_groups(hyperedges, original_to_bucket, bucket_count):
    coarse_hyperedges = []
    for he in hyperedges:
        mapped = []
        seen = set()
        for v in he:
            if 0 <= v < len(original_to_bucket):
                c = int(original_to_bucket[v])
                if c not in seen:
                    mapped.append(c)
                    seen.add(c)
        if len(mapped) > 1:
            coarse_hyperedges.append(mapped)
    return coarse_hyperedges


def _graph_to_hyperedges_from_clique(coarse_graph):
    if not coarse_graph.is_sparse:
        coarse_graph = coarse_graph.to_sparse()
    indices = coarse_graph.coalesce().indices().cpu().numpy()
    values = coarse_graph.coalesce().values().cpu().numpy()
    hyperedges = []
    seen = set()
    for idx in range(indices.shape[1]):
        u = int(indices[0, idx])
        v = int(indices[1, idx])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        hyperedges.append([u, v])
    return hyperedges


def coarsen_kahypar_like(hyperedges, num_nodes, q=2, coarsen_to=10, verbose=False, seed=None, lsh_planes=4, lsh_tables=32, use_lsh=False):
    """Fast KaHyPar-like coarsening with batched heavy-edge matching.

    The implementation avoids a global per-pair priority queue. Instead it runs
    repeated matching rounds:
    - visit alive vertices in random order,
    - pick the best unmatched neighbor by heavy-edge rating,
    - contract all selected pairs in a batch using a remap array,
    - rebuild coarse hyperedges for the next round.
    """
    rng = np.random.default_rng(seed)

    if use_lsh:
        lsh_map, lsh_groups = _lsh_bucketize_vertices(
            hyperedges,
            num_nodes,
            target_buckets=max(1, int(coarsen_to) * 4),
            num_planes=lsh_planes,
            num_tables=lsh_tables,
            seed=seed,
            verbose=verbose,
        )
        if verbose:
            print(f"[kahypar_like] LSH pre-coarsen: {num_nodes} -> {len(lsh_groups)} buckets")
        current_hyperedges = _rebuild_hyperedges_from_groups(hyperedges, lsh_map, len(lsh_groups))
        current_groups = [list(g) for g in lsh_groups]
    else:
        current_hyperedges = [list(dict.fromkeys(he)) for he in hyperedges if len(set(he)) > 1]
        current_groups = [[i] for i in range(num_nodes)]

    current_n = len(current_groups)
    if current_n == 0:
        empty_graph = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (0, 0),
        ).coalesce()
        return {
            'coarse_graph': empty_graph,
            'coarse_node_weights': torch.empty((0,), dtype=torch.float32),
            'coarse_groups': [],
            'original_to_coarse': np.empty((0,), dtype=np.int64),
            'coarse_hyperedges': [],
            'initial_assignment': np.empty((0,), dtype=np.int64),
        }

    target_coarse_size = max(1, int(coarsen_to))

    def build_incidence(hyperedge_list, vertex_count):
        vertex_to_edges = [set() for _ in range(vertex_count)]
        edge_vertices = []
        edge_weights = []
        for eid, he in enumerate(hyperedge_list):
            verts = []
            seen = set()
            for v in he:
                if 0 <= v < vertex_count and v not in seen:
                    verts.append(int(v))
                    seen.add(int(v))
            if len(verts) > 1:
                edge_vertices.append(verts)
                edge_weights.append(1.0)
                for v in verts:
                    vertex_to_edges[v].add(len(edge_vertices) - 1)
        return vertex_to_edges, edge_vertices, edge_weights

    round_id = 0
    while current_n > target_coarse_size:
        round_id += 1
        alive = np.ones(current_n, dtype=bool)
        matched = np.zeros(current_n, dtype=bool)
        partner = np.full(current_n, -1, dtype=np.int64)
        vertex_to_edges, edge_vertices, edge_weights = build_incidence(current_hyperedges, current_n)

        order = rng.permutation(current_n)
        pair_count = 0

        for u in order:
            if not alive[u] or matched[u]:
                continue

            ratings = {}
            for eid in vertex_to_edges[u]:
                verts = edge_vertices[eid]
                if len(verts) < 2:
                    continue
                contrib = float(edge_weights[eid]) / float(len(verts) - 1)
                for v in verts:
                    if v != u and alive[v] and not matched[v]:
                        ratings[v] = ratings.get(v, 0.0) + contrib

            if not ratings:
                continue

            v = max(ratings.items(), key=lambda item: (item[1], -item[0]))[0]
            if ratings[v] <= 0.0 or matched[v] or not alive[v]:
                continue

            matched[u] = True
            matched[v] = True
            partner[u] = v
            partner[v] = u
            pair_count += 1

        if pair_count == 0:
            break

        remap = np.full(current_n + pair_count, -1, dtype=np.int64)
        new_groups = []
        new_id = 0
        used = np.zeros(current_n, dtype=bool)

        for u in range(current_n):
            if not alive[u] or used[u]:
                continue
            v = partner[u]
            if v != -1 and used[v]:
                continue
            if v != -1:
                used[u] = True
                used[v] = True
                remap[u] = new_id
                remap[v] = new_id
                new_groups.append(current_groups[u] + current_groups[v])
            else:
                used[u] = True
                remap[u] = new_id
                new_groups.append(current_groups[u])
            new_id += 1

        new_hyperedges = []
        for he in current_hyperedges:
            mapped = []
            seen = set()
            for v in he:
                nv = remap[v]
                if nv < 0 or nv in seen:
                    continue
                mapped.append(int(nv))
                seen.add(int(nv))
            if len(mapped) > 1:
                new_hyperedges.append(mapped)

        # if verbose:
        #     print(f"[matching] round={round_id} alive={current_n} pairs={pair_count} -> {new_id}")

        if new_id == current_n:
            break

        current_hyperedges = new_hyperedges
        current_groups = new_groups
        current_n = new_id

    coarse_groups = current_groups
    coarse_hyperedges = current_hyperedges
    coarse_index = {node: idx for idx, node in enumerate(range(len(coarse_groups)))}

    original_to_coarse = np.empty(num_nodes, dtype=np.int64)
    for idx, members in enumerate(coarse_groups):
        for member in members:
            if member < num_nodes:
                original_to_coarse[member] = idx

    coarse_graph = build_clique_expanded_graph(coarse_hyperedges, num_nodes=len(coarse_groups), normalize_weight=True)
    coarse_node_weights = torch.tensor([len(g) for g in coarse_groups], dtype=torch.float32)

    initial_assignment = greedy_initial_hypergraph_partition(
        coarse_hyperedges,
        coarse_node_weights.cpu().numpy(),
        q,
        hyperedge_weights=[1.0] * len(coarse_hyperedges),
        epsilon=0.03,
        seed=seed,
    )

    return {
        'coarse_graph': coarse_graph,
        'coarse_node_weights': coarse_node_weights,
        'coarse_groups': coarse_groups,
        'original_to_coarse': original_to_coarse,
        'coarse_hyperedges': coarse_hyperedges,
        'initial_assignment': initial_assignment,
    }


def coarsen_fem_refine_kahypar(
    hyperedges,
    num_nodes,
    q=2,
    coarsen_to=10,
    num_trials=1,
    num_steps=10,
    dev='cpu',
    verbose=False,
    lsh_planes=4,
    lsh_tables=32,
    fem_mode='fem_as_hem',
):
    """FEM-assisted coarsening with two submodes.

    fem_as_hem:
        Use a sparse edge-variable QUBO to replace the HEM contraction step.
    fem_as_greedy_init:
        Use the normal KaHyPar-like coarsening pipeline, then replace the
        initial greedy coarse partition with a FEM-based initial partition.
    """
    from FEM import FEM as _FEM

    if fem_mode not in ('fem_as_hem', 'fem_as_greedy_init'):
        raise ValueError(f"Unknown fem_mode: {fem_mode}")

    def _fem_initial_partition(coarse_graph, coarse_node_weights):
        coarse_coupling = coarse_graph
        if not coarse_coupling.is_sparse:
            coarse_coupling = coarse_coupling.to_sparse()
        coarse_coupling = coarse_coupling.coalesce()

        num_coarse_nodes = int(coarse_coupling.shape[0])
        num_interactions = int(coarse_coupling._nnz() // 2)

        fem = _FEM.from_couplings(
            'bmincut',
            num_coarse_nodes,
            num_interactions,
            coarse_coupling,
            node_weights=coarse_node_weights,
        )
        fem.set_up_solver(num_trials, num_steps, dev=dev, q=max(2, int(q)))
        configs, results = fem.solve()
        best_idx = int(torch.argmin(results).item())
        return configs[best_idx].argmax(dim=1).cpu().numpy().astype(np.int64)

    if fem_mode == 'fem_as_greedy_init':
        res = coarsen_kahypar_like(
            hyperedges,
            num_nodes,
            q=q,
            coarsen_to=coarsen_to,
            verbose=verbose,
            seed=0,
            lsh_planes=lsh_planes,
            lsh_tables=lsh_tables,
            use_lsh=False,
        )
        res['initial_assignment'] = _fem_initial_partition(res['coarse_graph'], res['coarse_node_weights'])
        return res

    rng = np.random.default_rng(0)
    target_coarse = max(1, int(coarsen_to))

    # Parameters to bound QUBO size and penalties.
    max_qubo_vars = 20000
    conflict_scale = 1.2

    # current grouping: initially each original vertex is its own group
    current_groups = [[i] for i in range(num_nodes)]
    current_hyperedges = [list(dict.fromkeys(he)) for he in hyperedges if len(set(he)) > 1]
    current_n = len(current_groups)

    def build_pair_affinity(hyperedge_list, vertex_count):
        pair_aff = {}
        for he in hyperedge_list:
            verts = [int(v) for v in he if 0 <= v < vertex_count]
            verts = sorted(set(verts))
            if len(verts) <= 1:
                continue
            contrib = 1.0 / float(len(verts) - 1)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u = verts[i]
                    v = verts[j]
                    key = (u, v) if u < v else (v, u)
                    pair_aff[key] = pair_aff.get(key, 0.0) + contrib
        return pair_aff

    round_id = 0
    while current_n > target_coarse:
        round_id += 1
        pair_affinity = build_pair_affinity(current_hyperedges, current_n)
        if not pair_affinity:
            break

        # Pick top candidate pairs by affinity up to cap
        sorted_pairs = sorted(pair_affinity.items(), key=lambda kv: -kv[1])
        m = min(len(sorted_pairs), max(1, int(min(len(sorted_pairs), max_qubo_vars))))
        candidates = sorted_pairs[:m]

        var_index = {}
        vars_u = []
        vars_v = []
        weights = []
        for idx, ((u, v), w) in enumerate(candidates):
            var_index[(u, v)] = idx
            vars_u.append(int(u))
            vars_v.append(int(v))
            weights.append(float(w))

        m = len(weights)
        if m == 0:
            break

        max_w = max(weights)
        penalty = max(1e-3, max_w) * conflict_scale

        # Build sparse QUBO: diagonal h_i = -w_i (reward), conflicts J_ij = penalty
        q_rows = []
        q_cols = []
        q_vals = []

        # diagonals
        for i, w in enumerate(weights):
            q_rows.append(i)
            q_cols.append(i)
            q_vals.append(-float(w))

        # conflict pairs (edges sharing a vertex) -> positive quadratic penalty
        vertex_to_vars = {}
        for i, (u, v) in enumerate(zip(vars_u, vars_v)):
            vertex_to_vars.setdefault(u, []).append(i)
            vertex_to_vars.setdefault(v, []).append(i)

        # enumerate conflicts (i<j) and add J_ij to both symmetric positions half value
        conflicts_added = 0
        for var_list in vertex_to_vars.values():
            if len(var_list) <= 1:
                continue
            for a in range(len(var_list)):
                for b in range(a + 1, len(var_list)):
                    i = var_list[a]
                    j = var_list[b]
                    # add penalty to (i,j) and (j,i) as half-values so x^T Q x sums to J_ij x_i x_j
                    q_rows.extend([i, j])
                    q_cols.extend([j, i])
                    q_vals.extend([penalty / 2.0, penalty / 2.0])
                    conflicts_added += 1

        Q_sparse = torch.sparse_coo_tensor(
            torch.tensor([q_rows, q_cols], dtype=torch.long) if q_rows else torch.empty((2, 0), dtype=torch.long),
            torch.tensor(q_vals, dtype=torch.float32) if q_vals else torch.empty((0,), dtype=torch.float32),
            (m, m),
        ).coalesce()

        # FEM expected and inference functions for binary selection variables
        def _extract_binary_state(p: torch.Tensor) -> torch.Tensor:
            if p.dim() == 3:
                if p.shape[-1] != 2:
                    raise ValueError(f"Unexpected p shape for edge-QUBO: {tuple(p.shape)}")
                return p[..., 1]
            if p.dim() == 2:
                if p.shape[1] == 2 and (p.shape[0] % m) == 0:
                    return p.reshape(-1, m, 2)[..., 1]
                if p.shape[1] == m:
                    return p
            if p.dim() == 1:
                return p.unsqueeze(0)
            raise ValueError(f"Unexpected p shape for edge-QUBO: {tuple(p.shape)}")

        def expected_qubo(_, p: torch.Tensor) -> torch.Tensor:
            x = _extract_binary_state(p)
            Q = Q_sparse.to(device=p.device, dtype=p.dtype)
            pair_energy = (torch.sparse.mm(Q, x.T).T * x).sum(dim=1)
            return pair_energy

        def inference_qubo(_, p: torch.Tensor):
            x = _extract_binary_state(p)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            labels = (x >= 0.5).to(torch.long)
            config = torch.zeros((labels.shape[0], labels.shape[1], 2), dtype=p.dtype, device=p.device)
            config[..., 1] = labels.to(p.dtype)
            config[..., 0] = 1.0 - config[..., 1]
            return config, torch.zeros(config.shape[0], device=config.device)

        fem = _FEM()
        dummy = torch.zeros((m, m), dtype=torch.float32)
        fem.set_up_problem(m, 0, 'customize', dummy, customize_expected_func=expected_qubo, customize_infer_func=inference_qubo)
        fem.set_up_solver(max(1, num_trials), max(10, num_steps), anneal='lin', dev=dev, q=2, manual_grad=False)
        try:
            config, result = fem.solve()
        except Exception:
            # fallback to greedy matching if FEM fails
            selected = []
            for i, ((u, v), w) in enumerate(candidates):
                selected.append(i)
        else:
            best_idx = int(torch.argmin(result).item())
            chosen = config[best_idx].argmax(dim=1).cpu().numpy().astype(np.int64)
            selected = [i for i, val in enumerate(chosen) if val == 1]

        # Greedily apply non-conflicting contractions from selected set, highest weight first
        selected_sorted = sorted(selected, key=lambda i: -weights[i])
        used = [False] * current_n
        remap = np.full(current_n + len(selected_sorted), -1, dtype=np.int64)
        new_groups = []
        new_id = 0

        for i in selected_sorted:
            u = vars_u[i]
            v = vars_v[i]
            if used[u] or used[v]:
                continue
            used[u] = True
            used[v] = True
            remap[u] = new_id
            remap[v] = new_id
            new_groups.append(current_groups[u] + current_groups[v])
            new_id += 1

        # remaining singletons
        for u in range(current_n):
            if not used[u]:
                remap[u] = new_id
                new_groups.append(current_groups[u])
                new_id += 1

        # rebuild hyperedges
        new_hyperedges = []
        for he in current_hyperedges:
            mapped = []
            seen = set()
            for v in he:
                nv = remap[v]
                if nv < 0 or nv in seen:
                    continue
                mapped.append(int(nv))
                seen.add(int(nv))
            if len(mapped) > 1:
                new_hyperedges.append(mapped)

        if new_id == current_n:
            break

        current_groups = new_groups
        current_hyperedges = new_hyperedges
        current_n = new_id

    # finalize outputs
    coarse_groups = current_groups
    original_to_coarse = np.empty(num_nodes, dtype=np.int64)
    for idx, members in enumerate(coarse_groups):
        for member in members:
            if member < num_nodes:
                original_to_coarse[member] = idx

    coarse_hyperedges = _build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)
    coarse_graph = build_clique_expanded_graph(coarse_hyperedges, num_nodes=len(coarse_groups), normalize_weight=True)
    coarse_node_weights = torch.tensor([len(g) for g in coarse_groups], dtype=torch.float32)
    initial_assignment = greedy_initial_hypergraph_partition(
        coarse_hyperedges,
        coarse_node_weights.cpu().numpy(),
        q,
        hyperedge_weights=[1.0] * len(coarse_hyperedges),
        epsilon=0.03,
        seed=None,
    )
    return {
        'coarse_graph': coarse_graph,
        'coarse_node_weights': coarse_node_weights,
        'coarse_groups': coarse_groups,
        'original_to_coarse': original_to_coarse,
        'coarse_hyperedges': coarse_hyperedges,
        'initial_assignment': initial_assignment,
    }


def evaluate_coarse_cut(coarse_hyperedges, assignment):
    cut, imb = evaluate_kahypar_cut_value(np.asarray(assignment, dtype=int), coarse_hyperedges, hyperedge_weights=[1.0] * len(coarse_hyperedges))
    return cut, imb