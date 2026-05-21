
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
        config = torch.zeros_like(p)
        config.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
        # return dummy low objective values to allow FEM to just pick the best config based on argmax.
        dummy_results = torch.zeros(p.shape[0], device=p.device)
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


def simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, someflag=False, arg7=0, arg8=0, part=None, max_passes=10):
    """
    Simple replacement for kaffpa: perform local greedy refinement (KL/FM-like)
    starting from `part` (list of length n). Returns (edgecut, part_list).
    - `vwgt`, `xadj`, `adjcwgt`, `adjncy` follow KaHIP conventions.
    - `q` number of partitions.
    - `epsilon` allowed imbalance fraction.
    - `max_passes` number of refinement passes.
    """
    import numpy as _np

    n = len(vwgt)
    if part is None:
        # balanced random start
        base = _np.arange(n) % q
        _np.random.shuffle(base)
        part = base.tolist()
    else:
        part = list(part)

    # build neighbor lists with weights
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        start = xadj[i]
        end = xadj[i+1]
        for idx in range(start, end):
            j = adjncy[idx]
            w = adjcwgt[idx]
            neighbors[i].append((j, w))

    counts = _np.bincount(_np.array(part, dtype=int), minlength=q)
    ideal = n / float(q)
    max_size = ideal * (1.0 + epsilon)

    def compute_edgecut(parts):
        cut = 0.0
        for i in range(n):
            for j, w in neighbors[i]:
                if i < j and parts[i] != parts[j]:
                    cut += w
        return int(cut)

    for _pass in range(max_passes):
        moved = False
        order = _np.random.permutation(n)
        for v in order:
            old = part[v]
            # compute weight to each group
            weight_to = _np.zeros(q, dtype=float)
            for u, w in neighbors[v]:
                weight_to[part[u]] += w

            best_delta = 0.0
            best_group = old
            for g in range(q):
                if g == old:
                    continue
                if counts[g] + 1 > max_size:
                    continue
                # delta = change_in_cut = sum_w_to_old - sum_w_to_new
                delta = weight_to[old] - weight_to[g]
                if delta < best_delta:
                    best_delta = delta
                    best_group = g

            if best_group != old:
                # apply move
                part[v] = int(best_group)
                counts[old] -= 1
                counts[best_group] += 1
                moved = True

        if not moved:
            break

    edgecut = compute_edgecut(part)
    return int(edgecut), part


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
