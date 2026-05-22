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


def _lsh_bucketize_vertices(hyperedges, num_nodes, target_buckets=None, num_planes=12, num_tables=3, seed=None):
    """Pre-coarsen vertices with random-hyperplane LSH.

    Vertices hashed to the same bucket are treated as likely similar and are
    merged first. This is a preprocessing pass before the standard contraction
    routine.
    """
    rng = np.random.default_rng(seed)
    features = _vertex_feature_matrix(hyperedges, num_nodes)
    if num_nodes == 0:
        return np.arange(0, dtype=np.int64), []

    signatures = []
    tables = max(1, int(num_tables))
    planes_per_table = max(1, int(num_planes))
    if target_buckets is not None and num_nodes > 0:
        # Aim for a bucket count in the same ballpark as the requested coarse size.
        expected_bits = int(np.clip(np.ceil(np.log2(max(2, num_nodes / max(1, int(target_buckets))))), 4, 16))
        planes_per_table = max(planes_per_table, expected_bits)
        tables = max(tables, 1)

    for _ in range(tables):
        planes = rng.normal(size=(features.shape[1], planes_per_table)).astype(np.float32)
        proj = features @ planes
        bits = (proj >= 0.0).astype(np.uint8)
        bucket_ids = np.zeros(num_nodes, dtype=np.uint64)
        for bit_idx in range(bits.shape[1]):
            bucket_ids |= (bits[:, bit_idx].astype(np.uint64) << np.uint64(bit_idx))
        signatures.append(bucket_ids)

    buckets = {}
    for v in range(num_nodes):
        key = tuple(int(sig[v]) for sig in signatures)
        buckets.setdefault(key, []).append(v)

    if target_buckets is not None and len(buckets) < max(1, int(target_buckets)):
        # If LSH collapses too aggressively, split the largest buckets by vertex id
        # so the pre-coarsening does not destroy too much structure.
        bucket_items = sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
        while len(bucket_items) < max(1, int(target_buckets)) and bucket_items:
            key, verts = bucket_items.pop(0)
            if len(verts) <= 1:
                bucket_items.append((key, verts))
                break
            split_point = max(1, len(verts) // 2)
            left = verts[:split_point]
            right = verts[split_point:]
            bucket_items.append((key + ('L',), left))
            if right:
                bucket_items.append((key + ('R',), right))
        buckets = {k: v for k, v in bucket_items}

    groups = []
    for verts in buckets.values():
        if len(verts) == 1:
            groups.append([verts[0]])
        else:
            groups.append(list(verts))

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


def coarsen_kahypar_like(hyperedges, num_nodes, q=2, coarsen_to=10, verbose=False, seed=None, lsh_planes=8, lsh_tables=2):
    """Heap-based KaHyPar-like hypergraph contraction.

    The coarsener keeps contracting the best-rated pair until the coarse graph
    reaches `coarsen_to` or no valid pair remains. Pair ratings are defined as:
        sum_{e contains u,v} weight(e) / (|e| - 1)
    and stale heap entries are discarded lazily.
    """
    rng = np.random.default_rng(seed)
    _ = rng  # reserved for future community detection / tie-breaking

    # LSH preprocessing: bucket similar vertices together first.
    lsh_map, lsh_groups = _lsh_bucketize_vertices(
        hyperedges,
        num_nodes,
        target_buckets=max(1, int(coarsen_to) * 4),
        num_planes=lsh_planes,
        num_tables=lsh_tables,
        seed=seed,
    )
    if verbose:
        print(f"[kahypar_like] LSH pre-coarsen: {num_nodes} -> {len(lsh_groups)} buckets")

    pre_hyperedges = _rebuild_hyperedges_from_groups(hyperedges, lsh_map, len(lsh_groups))

    # dynamic coarse representation
    alive = {i: True for i in range(len(lsh_groups))}
    groups = {i: list(lsh_groups[i]) for i in range(len(lsh_groups))}
    next_node_id = len(lsh_groups)

    edge_vertices = {eid: set(he) for eid, he in enumerate(pre_hyperedges) if len(he) > 1}
    edge_weights = {eid: 1.0 for eid in edge_vertices}
    vertex_to_edges = {i: set() for i in range(len(lsh_groups))}
    for eid, verts in edge_vertices.items():
        for v in verts:
            if v < len(lsh_groups):
                vertex_to_edges.setdefault(v, set()).add(eid)

    pair_rating = {}
    heap = []

    for eid, verts in edge_vertices.items():
        verts_list = list(verts)
        if len(verts_list) < 2:
            continue
        contribution = float(edge_weights[eid]) / float(len(verts_list) - 1)
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                u, v = verts_list[i], verts_list[j]
                a, b = (u, v) if u < v else (v, u)
                pair_rating[(a, b)] = pair_rating.get((a, b), 0.0) + contribution

    for (u, v), rating in list(pair_rating.items()):
        _push_pair(heap, pair_rating, u, v, rating)

    def updateVertexPair(u, v):
        rating = _evaluate_pair_rating(u, v, alive, vertex_to_edges, edge_vertices, edge_weights)
        _push_pair(heap, pair_rating, u, v, rating)
        return rating

    def invalidate_vertex_pairs(vertex):
        for other in list(alive.keys()):
            if other == vertex or not alive.get(other, False):
                continue
            a, b = (vertex, other) if vertex < other else (other, vertex)
            if (a, b) in pair_rating:
                pair_rating.pop((a, b), None)

    def contract_pair(u, v):
        nonlocal next_node_id
        w = next_node_id
        next_node_id += 1

        groups[w] = groups.get(u, []) + groups.get(v, [])
        alive[u] = False
        alive[v] = False
        alive[w] = True

        incident_eids = set(vertex_to_edges.get(u, set())) | set(vertex_to_edges.get(v, set()))
        affected_vertices = set()

        for eid in incident_eids:
            verts = edge_vertices.get(eid)
            if not verts:
                continue
            if u not in verts and v not in verts:
                continue
            new_verts = set(verts)
            new_verts.discard(u)
            new_verts.discard(v)
            new_verts.add(w)
            edge_vertices[eid] = new_verts

            vertex_to_edges.setdefault(w, set()).add(eid)
            if u in vertex_to_edges:
                vertex_to_edges[u].discard(eid)
            if v in vertex_to_edges:
                vertex_to_edges[v].discard(eid)

            affected_vertices.update(new_verts)

        # Remove singleton hyperedges and keep edge maps compact.
        for eid, verts in list(edge_vertices.items()):
            if len(verts) <= 1:
                for x in list(verts):
                    if x in vertex_to_edges:
                        vertex_to_edges[x].discard(eid)
                edge_vertices.pop(eid, None)
                edge_weights.pop(eid, None)

        # Invalidate pairs touching u/v, then refresh only pairs touched by the contraction.
        for old in (u, v):
            for other in list(alive.keys()):
                if not alive.get(other, False) or other == old:
                    continue
                a, b = (old, other) if old < other else (other, old)
                pair_rating.pop((a, b), None)

        neighbor_vertices = set()
        for eid in vertex_to_edges.get(w, set()):
            for x in edge_vertices.get(eid, set()):
                if x != w and alive.get(x, False):
                    neighbor_vertices.add(x)

        for x in neighbor_vertices:
            updateVertexPair(w, x)

        # Refresh the pairs for vertices adjacent to the contraction boundary.
        for x in affected_vertices:
            if x == w or not alive.get(x, False):
                continue
            updateVertexPair(w, x)

        # if verbose:
            # print(f"[kahypar_like] contract ({u}, {v}) -> {w}, alive={sum(1 for x in alive if alive[x])}")

        return w

    while True:
        current_alive = [v for v in alive if alive[v]]
        if len(current_alive) <= max(1, int(coarsen_to)):
            break

        chosen = None
        while heap:
            neg_rating, u, v = heapq.heappop(heap)
            if not (alive.get(u, False) and alive.get(v, False)):
                continue
            key = (u, v)
            cur = pair_rating.get(key)
            if cur is None:
                continue
            if abs(cur + neg_rating) > 1e-12:
                continue
            chosen = (u, v, cur)
            break

        if chosen is None:
            break

        u, v, rating = chosen
        # matching-style contraction: immediately invalidate the pair and contract it.
        invalidate_vertex_pairs(u)
        invalidate_vertex_pairs(v)
        contract_pair(u, v)

    alive_nodes = [v for v in alive if alive[v]]
    coarse_groups = [groups[v] for v in alive_nodes]
    coarse_index = {node: idx for idx, node in enumerate(alive_nodes)}

    original_to_coarse = np.empty(num_nodes, dtype=np.int64)
    for idx, members in enumerate(coarse_groups):
        for member in members:
            if member < num_nodes:
                original_to_coarse[member] = idx

    coarse_hyperedges = []
    for verts in edge_vertices.values():
        mapped = []
        seen = set()
        for v in verts:
            if v in coarse_index:
                cv = coarse_index[v]
                if cv not in seen:
                    mapped.append(cv)
                    seen.add(cv)
        if len(mapped) > 1:
            coarse_hyperedges.append(mapped)

    coarse_graph = build_clique_expanded_graph(coarse_hyperedges, num_nodes=len(coarse_groups), normalize_weight=True)
    coarse_node_weights = torch.tensor([len(g) for g in coarse_groups], dtype=torch.float32)

    initial_assignment = greedy_initial_hypergraph_partition(
        coarse_hyperedges,
        len(coarse_groups),
        q,
        hyperedge_weights=[1.0] * len(coarse_hyperedges),
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


def coarsen_fem_refine_kahypar(hyperedges, num_nodes, q=2, coarsen_to=10, num_trials=1, num_steps=10, dev='cpu', verbose=False, lsh_planes=8, lsh_tables=2):
    """QUBO matching coarsening followed by the same coarse greedy initializer.

    The intent of this mode is to test the contraction stage and compare the
    resulting coarse cut against `kahypar_like` without running any external
    KaHyPar refinement.
    """
    from itertools import combinations

    from FEM.cyclic_expansion import solve_qubo_with_fem

    wprime = {}
    for he in hyperedges:
        verts = sorted(set(he))
        for u, v in combinations(verts, 2):
            key = (int(u), int(v))
            wprime[key] = wprime.get(key, 0.0) + 1.0

    candidate_edges = list(wprime.keys())
    if not candidate_edges:
        original_to_coarse = np.arange(num_nodes, dtype=np.int64)
        coarse_groups = [[i] for i in range(num_nodes)]
    else:
        s = len(candidate_edges)
        Q = np.zeros((s, s), dtype=float)
        w_vec = np.array([wprime[e] for e in candidate_edges], dtype=float)

        for i in range(s):
            Q[i, i] -= w_vec[i]

        P = max(1.0, float(w_vec.sum())) * 10.0
        incid = {v: [] for v in range(num_nodes)}
        for idx, (u, v) in enumerate(candidate_edges):
            incid[u].append(idx)
            incid[v].append(idx)

        for _, idxs in incid.items():
            for i in idxs:
                Q[i, i] += P - 2.0 * P
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = idxs[i]
                    b = idxs[j]
                    Q[a, b] += 2.0 * P
                    Q[b, a] += 2.0 * P

        assign = solve_qubo_with_fem(Q, num_trials=max(1, num_trials), num_steps=max(10, num_steps), dev=dev)

        matched = set()
        coarse_groups = []
        original_to_coarse = np.full(num_nodes, -1, dtype=np.int64)
        next_c = 0
        for idx, val in enumerate(assign):
            if int(val) == 1:
                u, v = candidate_edges[idx]
                if u in matched or v in matched:
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

    lsh_map, lsh_groups = _lsh_bucketize_vertices(
        hyperedges,
        num_nodes,
        target_buckets=max(1, int(coarsen_to) * 4),
        num_planes=lsh_planes,
        num_tables=lsh_tables,
    )
    coarse_groups = [list(g) for g in lsh_groups]
    pre_hyperedges = _rebuild_hyperedges_from_groups(hyperedges, lsh_map, len(lsh_groups))
    coarse_hyperedges = _build_coarse_hyperedges(pre_hyperedges, np.arange(len(lsh_groups), dtype=np.int64), len(lsh_groups))
    coarse_graph = build_clique_expanded_graph(coarse_hyperedges, num_nodes=len(coarse_groups), normalize_weight=True)
    coarse_node_weights = torch.tensor([len(g) for g in coarse_groups], dtype=torch.float32)

    initial_assignment = greedy_initial_hypergraph_partition(
        coarse_hyperedges,
        len(coarse_groups),
        q,
        hyperedge_weights=[1.0] * len(coarse_hyperedges),
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
