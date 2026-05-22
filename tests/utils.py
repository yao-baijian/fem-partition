
import numpy as np
import torch
from itertools import combinations

def parse_hypergraph_edges(instance_path: str) -> list:
    hyperedges = []
    try:
        with open(instance_path, 'r') as f:
            f.readline()
            for line in f:
                if line.strip():
                    vertices = [int(v) - 1 for v in line.split() if v.strip()]
                    if len(vertices) > 1:  
                        hyperedges.append(vertices)
        # print(f"Parsed {len(hyperedges)} hyperedges from {instance_path}")
        return hyperedges
    except Exception as e:
        print(f"Error parsing hypergraph: {e}")
        return []

def evaluate_cut_value(assignment: np.ndarray, hyperedges: list) -> int:
    cut_count = 0
    for hyperedge in hyperedges:
        groups_in_hyperedge = set()
        for vertex in hyperedge:
            if vertex < len(assignment):
                groups_in_hyperedge.add(assignment[vertex])
        

        if len(groups_in_hyperedge) > 1:
            cut_count += 1
    
    return cut_count

def evaluate_kahypar_cut_value(assignment: np.ndarray, hyperedges: list, hyperedge_weights: list = None) -> float:
    """
    sum_{e in cut} (λ(e) - 1) * w(e)
    """

    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)
    
    total_cut_value = 0
    
    for hyperedge, weight in zip(hyperedges, hyperedge_weights):
        groups_in_hyperedge = set()
        if len(hyperedge) > 1:
            for vertex in hyperedge:
                groups_in_hyperedge.add(assignment[vertex])
        lambda_e = len(groups_in_hyperedge)
        if lambda_e > 1:
            total_cut_value += (lambda_e - 1) * weight
    
    arr = np.asarray(assignment, dtype=int)
    q = int(arr.max()) + 1
    counts = np.bincount(arr, minlength=q)
    ideal = arr.size / float(q)
    imbalance_per_group = np.abs(counts - ideal) / ideal
    max_imbalance = float(np.max(imbalance_per_group))
    return total_cut_value, max_imbalance


def build_clique_expanded_graph(hyperedges: list, num_nodes: int = None, normalize_weight: bool = True):
    if num_nodes is None:
        num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1

    rows = []
    cols = []
    values = []

    for hyperedge in hyperedges:
        if len(hyperedge) < 2:
            continue
        edge_weight = 1.0 / (len(hyperedge) - 1) if normalize_weight else 1.0
        for u, v in combinations(hyperedge, 2):
            rows.extend([u, v])
            cols.extend([v, u])
            values.extend([edge_weight, edge_weight])

    if not rows:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (num_nodes, num_nodes)).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long)
    weights = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, weights, (num_nodes, num_nodes)).coalesce()


def _sparse_to_adjacency_dict(J: torch.Tensor):
    J = J.coalesce()
    n = J.shape[0]
    adjacency = [dict() for _ in range(n)]
    indices = J.indices()
    values = J.values()
    for idx in range(values.numel()):
        row = int(indices[0, idx])
        col = int(indices[1, idx])
        if row == col:
            continue
        adjacency[row][col] = adjacency[row].get(col, 0.0) + float(values[idx].item())
    return adjacency


def coarsen_graph_by_matching(J: torch.Tensor, node_weights=None, max_node_weight=None, coarsen_to: int = 500, max_rounds: int = 20):
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    groups = [[node] for node in range(n)]
    if node_weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.array(node_weights, dtype=np.float32)
        
    if max_node_weight is None:
        max_node_weight = max(weights.sum() / 50.0, np.max(weights) * 2)
        
    current_J = J
    current_n = n
    current_weights = weights
    
    for _ in range(max_rounds):
        if current_n <= coarsen_to:
            break
            
        adjacency = _sparse_to_adjacency_dict(current_J)
        
        matched = np.zeros(current_n, dtype=bool)
        remap = np.full(current_n, -1, dtype=np.int64)
        new_n = 0
        
        visit_order = np.random.permutation(current_n)
        
        new_groups = []
        new_weights = []
        
        for u in visit_order:
            if matched[u]:
                continue
            matched[u] = True
            
            best_v = -1
            best_w = -1.0
            for v, edge_w in adjacency[u].items():
                if not matched[v] and current_weights[u] + current_weights[v] <= max_node_weight:
                    if edge_w > best_w:
                        best_w = edge_w
                        best_v = v
                        
            if best_v != -1:
                matched[best_v] = True
                remap[u] = new_n
                remap[best_v] = new_n
                new_groups.append(groups[u] + groups[best_v])
                new_weights.append(current_weights[u] + current_weights[best_v])
            else:
                remap[u] = new_n
                new_groups.append(groups[u])
                new_weights.append(current_weights[u])
                
            new_n += 1
            
        if new_n == current_n:
            break
            
        indices = current_J.indices()
        values = current_J.values()
        
        coarse_rows = remap[indices[0].numpy()]
        coarse_cols = remap[indices[1].numpy()]
        
        valid = coarse_rows != coarse_cols
        
        if np.any(valid):
            coarse_indices = torch.tensor(np.stack([coarse_rows[valid], coarse_cols[valid]]), dtype=torch.long)
            coarse_values = values[torch.from_numpy(valid)]
            current_J = torch.sparse_coo_tensor(coarse_indices, coarse_values, (new_n, new_n)).coalesce()
        else:
            current_J = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (new_n, new_n)).coalesce()
            
        current_n = new_n
        current_weights = np.array(new_weights, dtype=np.float32)
        groups = new_groups
        
    coarse_node_weights = torch.tensor(current_weights, dtype=torch.float32)
    
    original_to_coarse = np.empty(n, dtype=np.int64)
    for c_node, members in enumerate(groups):
        for member in members:
            original_to_coarse[member] = c_node
            
    return current_J, coarse_node_weights, groups, original_to_coarse


def coarsen_graph_n_level(
    J: torch.Tensor,
    node_weights=None,
    max_node_weight=None,
    coarsen_to: int = 500,
    max_rounds: int = 1000,
    verbose: bool = False,
    log_every: int = 100,
):
    """
    n-level style coarsening: merge only one vertex pair per level.

    This intentionally creates many fine-grained levels so uncoarsening can
    remain cheap and local, similar to the high-level structure used by KaHyPar.
    """
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]

    if n == 0:
        empty_w = torch.empty((0,), dtype=torch.float32)
        return J, empty_w, [], np.empty((0,), dtype=np.int64)

    if node_weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.array(node_weights, dtype=np.float32)

    if max_node_weight is None:
        max_node_weight = max(weights.sum() / 50.0, np.max(weights) * 2)

    groups = [[node] for node in range(n)]
    current_J = J
    current_weights = weights
    current_n = n

    # Build an adjacency dict for fast updates and a candidate heap (max-heap via negative weights).
    import heapq

    adjacency = _sparse_to_adjacency_dict(current_J)
    alive = np.ones(current_n, dtype=bool)
    # Use lists for weights/groups that will be mutated in place.
    weights = list(current_weights.tolist())

    heap = []  # entries are (-edge_w, u, v)
    for u, nbrs in enumerate(adjacency):
        for v, edge_w in nbrs.items():
            if u < v and weights[u] + weights[v] <= max_node_weight:
                heapq.heappush(heap, (-float(edge_w), u, v))

    merges_done = 0
    while merges_done < max_rounds:
        # stop when target reached
        if sum(alive) <= coarsen_to:
            break

        # find a valid candidate
        best_pair = None
        while heap:
            neg_w, u, v = heapq.heappop(heap)
            if not (alive[u] and alive[v]):
                continue
            # validate current edge weight still exists and weight constraints
            cur_w = adjacency[u].get(v, 0.0)
            if cur_w <= 0.0:
                continue
            if weights[u] + weights[v] > max_node_weight:
                continue
            best_pair = (u, v, float(cur_w))
            break

        if best_pair is None:
            break

        u, v, w = best_pair

        # merge v into u (choose smaller id as representative for stability)
        if v < u:
            u, v = v, u

        # combine groups and weights
        groups[u] = groups[u] + groups[v]
        weights[u] = float(weights[u]) + float(weights[v])
        alive[v] = False

        # Merge adjacency of v into u
        for x, wvx in list(adjacency[v].items()):
            if x == u:
                continue
            # add weight to u-x
            adjacency[u][x] = adjacency[u].get(x, 0.0) + float(wvx)
            # replace v with u in x's adjacency
            adjacency[x][u] = adjacency[x].get(u, 0.0) + float(adjacency[x].pop(v, 0.0))

        # remove self-loops if any
        adjacency[u].pop(u, None)
        adjacency[v].clear()

        # push new candidate edges for updated u
        for nbr, ew in adjacency[u].items():
            if alive[nbr] and weights[u] + weights[nbr] <= max_node_weight:
                heapq.heappush(heap, (-float(ew), min(u, nbr), max(u, nbr)))

        merges_done += 1

        if verbose and (merges_done % max(1, int(log_every)) == 0):
            print(f"[coarsen_n_level] merges={merges_done}, alive={int(sum(alive))}")

    # compact remaining alive nodes to build the coarse sparse tensor
    old_to_new = np.full(current_n, -1, dtype=np.int64)
    new_groups = []
    new_weights = []
    next_idx = 0
    for i in range(current_n):
        if not alive[i]:
            continue
        old_to_new[i] = next_idx
        new_groups.append(groups[i])
        new_weights.append(weights[i])
        next_idx += 1

    if next_idx == 0:
        current_J = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (0, 0)).coalesce()
    else:
        rows = []
        cols = []
        vals = []
        for u in range(current_n):
            if old_to_new[u] == -1:
                continue
            for v, ew in adjacency[u].items():
                if old_to_new[v] == -1:
                    continue
                r = old_to_new[u]
                c = old_to_new[v]
                if r != c:
                    rows.append(r)
                    cols.append(c)
                    vals.append(float(ew))

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
            current_J = torch.sparse_coo_tensor(indices, values, (next_idx, next_idx)).coalesce()
        else:
            current_J = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (next_idx, next_idx)).coalesce()

    current_weights = np.array(new_weights, dtype=np.float32)
    groups = new_groups
    current_n = next_idx

    coarse_node_weights = torch.tensor(current_weights, dtype=torch.float32)
    original_to_coarse = np.empty(n, dtype=np.int64)
    for c_node, members in enumerate(groups):
        for member in members:
            original_to_coarse[member] = c_node

    return current_J, coarse_node_weights, groups, original_to_coarse


def expand_coarse_labels(coarse_groups: list, coarse_labels: np.ndarray, num_nodes: int):
    labels = np.empty(num_nodes, dtype=np.int64)
    for coarse_node, members in enumerate(coarse_groups):
        for member in members:
            labels[member] = coarse_labels[coarse_node]
    return labels

class PUBOObjective:
    def __init__(self, hyperedges, hyperedge_weights, q, num_nodes, node_weights, imbalance_weight=5.0, obj_type='cut_net', max_degree=5):
        import torch
        from FEM.problem import weighted_imbalance_penalty
        self.groups = {}
        for size in range(2, max_degree + 1):
            self.groups[size] = {'indices': [], 'weights': []}
            
        large_he = []
        large_weights = []
        # Calculate degrees
        node_degrees = np.ones(num_nodes, dtype=np.float32) # Add 1 to avoid div by zero
        for he, w in zip(hyperedges, hyperedge_weights):
            for v in he:
                if v < num_nodes:
                    node_degrees[v] += w

            if len(he) <= max_degree:
                self.groups[len(he)]['indices'].append(he)
                self.groups[len(he)]['weights'].append(w)
            else:
                large_he.append(he)
                large_weights.append(w)
                
        self.tensors_by_size = {}
        for size, data in self.groups.items():
            if data['indices']:
                self.tensors_by_size[size] = {
                    'idx': torch.tensor(data['indices'], dtype=torch.long),
                    'weight': torch.tensor(data['weights'], dtype=torch.float32)
                }
                
        if large_he:
            clique_J = build_clique_expanded_graph(large_he, num_nodes=num_nodes, normalize_weight=True)
            self.clique_J = clique_J.to_dense()
        else:
            self.clique_J = None
            
        self.node_weights = torch.tensor(node_weights, dtype=torch.float32) if node_weights is not None else torch.ones(num_nodes)
        self.node_degrees = torch.tensor(node_degrees, dtype=torch.float32)
        self.imbalance_weight = imbalance_weight
        self.obj_type = obj_type
        self.weighted_imbalance_penalty = weighted_imbalance_penalty
        self.q = q
        
    def to(self, dev):
        for size in self.tensors_by_size:
            self.tensors_by_size[size]['idx'] = self.tensors_by_size[size]['idx'].to(dev)
            self.tensors_by_size[size]['weight'] = self.tensors_by_size[size]['weight'].to(dev)
        if self.clique_J is not None:
            self.clique_J = self.clique_J.to(dev)
        self.node_weights = self.node_weights.to(dev)
        self.node_degrees = self.node_degrees.to(dev)

    def expectation(self, _, p):
        # Optional: gradient scaling & clipping hook on p
        if p.requires_grad and not hasattr(p, 'pubo_hook_registered'):
            def scale_and_clip(grad):
                # Node degree normalization
                g = grad / self.node_degrees.view(1, -1, 1)
                # Gradient clipping
                g = torch.clamp(g, -5.0, 5.0)
                return g
            p.register_hook(scale_and_clip)
            p.pubo_hook_registered = True

        dev = p.device
        self.to(dev)
        
        loss = 0.0
        
        for size, t in self.tensors_by_size.items():
            idx = t['idx'] 
            weight = t['weight'] 
            
            p_e = p[:, idx, :] 
            
            if self.obj_type == 'cut_net':
                prod = p_e.prod(dim=2) 
                sum_prod = prod.sum(dim=2) 
                term = weight * (1.0 - sum_prod)
                loss = loss + term.sum(dim=1)
            elif self.obj_type == 'km1':
                prod = (1.0 - p_e).prod(dim=2) 
                sum_term = (1.0 - prod).sum(dim=2) 
                term = weight * (sum_term - 1.0)
                loss = loss + term.sum(dim=1)
                
        if self.clique_J is not None:
            clique_loss = ((self.clique_J @ p) * (1 - p)).sum(dim=(1, 2))
            loss = loss + clique_loss
            
        imb_penalty = self.weighted_imbalance_penalty(p, self.node_weights.cpu().numpy())
        loss = loss + self.imbalance_weight * imb_penalty
        
        return loss

    def inference(self, _, p):
        import torch
        # Dummy result since we recalculate cut with `evaluate_kahypar_cut_value` anyway. 
        # But FEM solver needs `config` and `results`.
        q = int(self.q)
        n = int(self.node_weights.numel())

        if p.dim() == 2:
            # FEM may flatten the batch dimension into the first axis, keep q on the last axis.
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
        # return dummy low objective values to allow FEM to just pick the best config based on argmax.
        dummy_results = torch.zeros(config.shape[0], device=p.device)
        return config, dummy_results

def greedy_refine_hypergraph(
    assignment: np.ndarray, 
    hyperedges: list, 
    hyperedge_weights: list, 
    q: int, 
    max_passes: int = 5,
    max_imbalance: float = 0.05
) -> np.ndarray:
    assignment = assignment.copy()
    num_nodes = len(assignment)
    
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)
        
    he_pins = [np.zeros(q, dtype=np.int32) for _ in range(len(hyperedges))]
    node_to_he = [[] for _ in range(num_nodes)]
    
    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)
                
    group_sizes = np.bincount(assignment, minlength=q)
    ideal_size = num_nodes / float(q)
    max_size = ideal_size * (1.0 + max_imbalance)
    
    for pass_idx in range(max_passes):
        moved_any = False
        nodes = np.arange(num_nodes)
        np.random.shuffle(nodes)
        
        for v in nodes:
            old_group = assignment[v]
            
            best_gain = 0.0
            best_group = old_group
            
            for new_group in range(q):
                if new_group == old_group:
                    continue
                    
                if group_sizes[new_group] + 1 > max_size:
                    continue
                    
                gain = 0.0
                for e_idx in node_to_he[v]:
                    pins = he_pins[e_idx]
                    weight = hyperedge_weights[e_idx]
                    
                    if pins[old_group] == 1:
                        gain += weight
                        
                    if pins[new_group] == 0:
                        gain -= weight
                        
                if gain > best_gain:
                    best_gain = gain
                    best_group = new_group
                    
            if best_group != old_group:
                assignment[v] = best_group
                group_sizes[old_group] -= 1
                group_sizes[best_group] += 1
                
                for e_idx in node_to_he[v]:
                    he_pins[e_idx][old_group] -= 1
                    he_pins[e_idx][best_group] += 1
                    
                moved_any = True
                
        if not moved_any:
            break
            
    return assignment


def greedy_refine_hypergraph_incremental(
    assignment: np.ndarray,
    hyperedges: list,
    hyperedge_weights: list,
    q: int,
    max_passes: int = 5,
    max_imbalance: float = 0.05,
):
    """
    Incremental local refinement that only re-evaluates affected vertices
    (the moved vertex and its L1 hypergraph neighbors).
    """
    assignment = assignment.copy()
    num_nodes = len(assignment)

    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)

    he_pins = [np.zeros(q, dtype=np.int32) for _ in range(len(hyperedges))]
    node_to_he = [[] for _ in range(num_nodes)]
    vertex_neighbors = [set() for _ in range(num_nodes)]

    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)
        for u in he:
            if u < num_nodes:
                for v in he:
                    if u != v and v < num_nodes:
                        vertex_neighbors[u].add(v)

    group_sizes = np.bincount(assignment, minlength=q)
    ideal_size = num_nodes / float(q)
    max_size = ideal_size * (1.0 + max_imbalance)

    active = np.zeros(num_nodes, dtype=bool)
    queue = list(np.where(np.ones(num_nodes, dtype=bool))[0])

    def move_gain(v, new_group):
        old_group = assignment[v]
        if new_group == old_group:
            return 0.0
        gain = 0.0
        for e_idx in node_to_he[v]:
            pins = he_pins[e_idx]
            weight = hyperedge_weights[e_idx]
            if pins[old_group] == 1:
                gain += weight
            if pins[new_group] == 0:
                gain -= weight
        return gain

    for _pass in range(max_passes):
        moved_any = False
        # Only revisit vertices that are active or whose neighbors were affected.
        while queue:
            v = queue.pop()
            active[v] = False

            old_group = assignment[v]
            best_gain = 0.0
            best_group = old_group

            for new_group in range(q):
                if new_group == old_group:
                    continue
                if group_sizes[new_group] + 1 > max_size:
                    continue
                gain = move_gain(v, new_group)
                if gain > best_gain:
                    best_gain = gain
                    best_group = new_group

            if best_group != old_group:
                assignment[v] = best_group
                group_sizes[old_group] -= 1
                group_sizes[best_group] += 1

                for e_idx in node_to_he[v]:
                    he_pins[e_idx][old_group] -= 1
                    he_pins[e_idx][best_group] += 1

                moved_any = True

                affected = set(vertex_neighbors[v])
                affected.add(v)
                for u in affected:
                    if not active[u]:
                        queue.append(u)
                        active[u] = True

        if not moved_any:
            break

        # Re-seed the frontier for the next pass with boundary vertices only.
        frontier = set()
        for v in range(num_nodes):
            for e_idx in node_to_he[v]:
                pins = he_pins[e_idx]
                if pins[assignment[v]] == 1:
                    frontier.add(v)
                    frontier.update(vertex_neighbors[v])
                    break
        queue = list(frontier)
        for v in queue:
            active[v] = True

    return assignment


def greedy_initial_hypergraph_partition(
    hyperedges: list,
    num_nodes: int,
    q: int,
    hyperedge_weights: list = None,
    max_imbalance: float = 0.05,
    seed: int = None,
):
    """
    Build a balanced initial q-way partition for a hypergraph using a simple
    greedy vertex placement heuristic.

    The goal is not to mimic KaHyPar's full machinery, but to provide a
    deterministic, self-contained coarse assignment that can be refined later.
    """
    rng = np.random.default_rng(seed)
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)

    node_to_he = [[] for _ in range(num_nodes)]
    node_weight = np.zeros(num_nodes, dtype=float)
    for e_idx, he in enumerate(hyperedges):
        w = float(hyperedge_weights[e_idx])
        for v in he:
            if 0 <= v < num_nodes:
                node_to_he[v].append(e_idx)
                node_weight[v] += w

    order = np.arange(num_nodes)
    # High-degree / high-weight vertices first; break ties randomly.
    tie_breaker = rng.random(num_nodes)
    order = np.lexsort((tie_breaker, -node_weight))

    assignment = np.full(num_nodes, -1, dtype=np.int64)
    group_sizes = np.zeros(q, dtype=np.int64)
    ideal = num_nodes / float(q)
    max_size = int(np.ceil(ideal * (1.0 + max_imbalance)))

    if num_nodes >= q:
        seed_nodes = order[:q]
        for g, v in enumerate(seed_nodes):
            assignment[v] = g
            group_sizes[g] += 1
        remaining_order = order[q:]
    else:
        remaining_order = order

    def boundary_cost(v, g):
        cost = 0.0
        for e_idx in node_to_he[v]:
            he = hyperedges[e_idx]
            w = float(hyperedge_weights[e_idx])
            pins = 0
            same = 0
            for u in he:
                au = assignment[u]
                if au != -1:
                    pins += 1
                    if au == g:
                        same += 1
            # Prefer groups where the new vertex joins existing pins.
            if same == 0:
                cost += w
            elif same == pins:
                cost -= w
        return cost

    for v in remaining_order:
        best_group = None
        best_cost = None
        candidates = np.arange(q)
        rng.shuffle(candidates)
        for g in candidates:
            if group_sizes[g] + 1 > max_size:
                continue
            cost = boundary_cost(v, g)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_group = g

        if best_group is None:
            best_group = int(np.argmin(group_sizes))

        assignment[v] = best_group
        group_sizes[best_group] += 1

    return assignment


def simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, someflag=False, arg7=0, arg8=0, part=None, max_passes=10):
    """
    Simple replacement for kaffpa: perform FM-style local refinement starting
    from `part` (list of length n).

    This implementation uses:
    - a max-heap of candidate single-vertex moves,
    - lazy gain recomputation,
    - per-pass locking,
    - rollback to the best prefix of moves.

    Returns (edgecut, part_list).
    """
    import heapq
    import numpy as _np

    n = len(vwgt)
    if part is None:
        base = _np.arange(n) % q
        _np.random.shuffle(base)
        part = base.tolist()
    else:
        part = [int(x) for x in part]

    # Build symmetric adjacency lists. If the input contains duplicate edges,
    # we preserve them and aggregate weights when computing gains.
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for idx in range(xadj[i], xadj[i + 1]):
            j = int(adjncy[idx])
            w = float(adjcwgt[idx])
            if j != i:
                neighbors[i].append((j, w))

    def edgecut_of(parts):
        cut = 0.0
        for i in range(n):
            pi = parts[i]
            for j, w in neighbors[i]:
                if i < j and pi != parts[j]:
                    cut += w
        return int(round(cut))

    def best_destination(vertex, parts):
        old = parts[vertex]
        weight_to = _np.zeros(q, dtype=float)
        for nbr, w in neighbors[vertex]:
            weight_to[parts[nbr]] += w

        best_group = old
        best_delta = 0.0
        for g in range(q):
            if g == old:
                continue
            # moving v from old -> g changes cut by w_to_old - w_to_g
            delta = weight_to[old] - weight_to[g]
            if delta < best_delta:
                best_delta = delta
                best_group = g
        return best_group, float(best_delta)

    def feasible_move(group_sizes, old_group, new_group, max_size):
        return group_sizes[new_group] + 1 <= max_size

    counts = _np.bincount(_np.asarray(part, dtype=int), minlength=q).astype(int)
    ideal = n / float(q)
    max_size = ideal * (1.0 + float(epsilon))

    for _pass in range(max_passes):
        locked = _np.zeros(n, dtype=bool)
        current_parts = part[:]
        current_counts = counts.copy()
        pass_start_cut = edgecut_of(current_parts)
        current_cut = pass_start_cut
        best_cut = current_cut
        best_state = current_parts[:]

        # Bucket-queue FM-style implementation for faster selection
        # Discretize gains to integer buckets using a scale factor
        scale = 1000.0

        buckets = {}  # gain_score -> list of vertices
        vertex_target = {}  # v -> target group

        def gain_key(delta):
            # higher key == better move (more negative delta)
            return int(round(-delta * scale))

        def insert_vertex(v):
            if locked[v]:
                return
            g, delta = best_destination(v, current_parts)
            if g == current_parts[v] or not feasible_move(current_counts, current_parts[v], g, max_size):
                return
            k = gain_key(delta)
            buckets.setdefault(k, []).append(v)
            vertex_target[v] = g

        for v in range(n):
            insert_vertex(v)

        moved = False
        # Maintain a sorted list of keys lazily when needed
        while buckets:
            # get current best key
            best_k = max(buckets.keys())
            # pop a vertex from that bucket
            v = buckets[best_k].pop()
            if not buckets[best_k]:
                del buckets[best_k]

            if locked[v]:
                vertex_target.pop(v, None)
                continue

            # Recompute lazy
            best_g, best_delta = best_destination(v, current_parts)
            k_new = gain_key(best_delta)
            if best_g != vertex_target.get(v, None) or k_new != best_k:
                # stale entry; reinsert if still valid
                vertex_target[v] = best_g
                if best_g != current_parts[v] and feasible_move(current_counts, current_parts[v], best_g, max_size):
                    buckets.setdefault(k_new, []).append(v)
                else:
                    vertex_target.pop(v, None)
                continue

            g = best_g
            delta = best_delta
            if g == current_parts[v] or not feasible_move(current_counts, current_parts[v], g, max_size):
                locked[v] = True
                vertex_target.pop(v, None)
                continue

            # perform move
            old = current_parts[v]
            current_parts[v] = g
            current_counts[old] -= 1
            current_counts[g] += 1
            locked[v] = True
            moved = True
            vertex_target.pop(v, None)

            current_cut += int(round(delta))
            if current_cut < best_cut:
                best_cut = current_cut
                best_state = current_parts[:]

            # neighbors gains changed; reinsert them
            for nbr, _w in neighbors[v]:
                if not locked[nbr]:
                    insert_vertex(nbr)

        if not moved:
            break

        part = best_state
        counts = _np.bincount(_np.asarray(part, dtype=int), minlength=q).astype(int)

        # Stop if the pass did not improve over the cut at the start of the pass.
        if best_cut >= pass_start_cut:
            break

    return edgecut_of(part), part


def call_pymetis_with_part(q, adjacency_list, part=None):
    """Call pymetis.part_graph and pass `part` when supported by the wrapper.
    If the wrapper doesn't accept `part`, prints a clear warning and calls
    without it (no silent fallback). Returns (edgecuts, parts).
    """
    import importlib, inspect, sys
    try:
        pymetis = importlib.import_module('pymetis')
    except Exception as e:
        raise ImportError(f"pymetis is not available: {e}")

    try:
        sig = inspect.signature(pymetis.part_graph)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    if 'part' in params:
        return pymetis.part_graph(q, adjacency=adjacency_list, part=part)
    else:
        # Explicit informative warning (not silent fallback)
        print("Warning: installed pymetis.part_graph does not accept 'part' argument; calling without initial partition", file=sys.stderr)
        return pymetis.part_graph(q, adjacency=adjacency_list)
