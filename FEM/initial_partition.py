"""FEM-based initial partitioning utilities for coarse graphs.

Provides routines to build Ising and QUBO matrices for k=2 partitioning
and a wrapper `fem_initial_partition` that uses the local `FEM` solver
to optimize the QUBO in a mean-field (product distribution) sense and
return a discrete initial partition suitable for uncoarsening.

API:
  build_ising_A(n, edges, weights, c, lambda_penalty) -> A (n x n)
  build_qubo_from_A(A) -> Q (n x n)   # QUBO matrix for binary x in {0,1}
  fem_initial_partition(W, c, k=2, lambda_penalty=1.0, fem_opts={}) -> np.ndarray

The returned assignment is a 1D numpy array of integers in {0,1,...,k-1}
with length n_coarse, compatible with `expand_coarse_labels` used
elsewhere in the codebase.
"""

from typing import Sequence, Tuple, Optional
import numpy as np
import torch


def build_ising_A(n: int, edges: Sequence[Tuple[int,int]], weights: Sequence[float], c: np.ndarray, lambda_penalty: float) -> np.ndarray:
    """Build symmetric Ising pairwise coefficient matrix A for k=2.

    H(σ) = sum_{i<j} A_ij σ_i σ_j,
    with A_ij = -w_ij + lambda_penalty * c_i * c_j (for i!=j) and A_ii = 0.

    Args:
      n: number of coarse nodes
      edges: iterable of (i,j) indices for edges (undirected)
      weights: iterable of edge weights (same order as edges)
      c: node weights array shape (n,)
      lambda_penalty: penalty multiplier for balance term

    Returns:
      A: (n,n) numpy array symmetric, zero diagonal.
    """
    A = np.zeros((n, n), dtype=float)
    # add graph contribution -w_ij
    for (i, j), w in zip(edges, weights):
        if i == j:
            continue
        A[i, j] += -w
        A[j, i] += -w

    # add balance coupling lambda * c_i * c_j
    c = np.asarray(c, dtype=float).reshape(-1)
    outer = lambda_penalty * np.outer(c, c)
    # ensure diagonal remains zero (we only want pairwise coupling)
    A += outer
    np.fill_diagonal(A, 0.0)
    return A


def build_qubo_from_A(A: np.ndarray) -> np.ndarray:
    """Convert Ising pairwise matrix A (H = sum_{i<j} A_ij σ_i σ_j)
    into a QUBO matrix Q for binary variables x in {0,1} using σ = 2x - 1.

    Derivation (sketch):
      σ_i σ_j = 4 x_i x_j - 2 x_i - 2 x_j + 1.
      H = sum_{i<j} A_ij (4 x_i x_j -2 x_i -2 x_j +1)
      Rearranging yields an objective of the form x^T Q x + const,
      with off-diagonal Q_ij = 2 * A_ij (i != j) and diagonal Q_ii = -2 * sum_{j != i} A_ij.

    Args:
      A: symmetric (n,n) numpy array with zero diagonal.

    Returns:
      Q: symmetric (n,n) numpy array representing the QUBO such that
         energy(x) = x^T Q x + const for binary x in {0,1}^n.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=float)
    # off-diagonal
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] = 2.0 * A[i, j]
            Q[j, i] = Q[i, j]

    # diagonal terms to capture linear coefficients (x_i^2 = x_i)
    for i in range(n):
        Q[i, i] = -2.0 * np.sum(A[i, :])

    return Q


def fem_initial_partition(coarse_adj_matrix: Optional[np.ndarray], edges: Optional[Sequence[Tuple[int,int]]], edge_weights: Optional[Sequence[float]], c: np.ndarray, k: int = 2, lambda_penalty: float = 1.0, num_trials: int = 8, num_steps: int = 200, dev: str = 'cpu') -> np.ndarray:
    """Run FEM to obtain an initial partition on the coarse graph for k=2.

    This function supports two input styles for the coarse graph:
      - provide `coarse_adj_matrix` as an (n,n) dense / numpy array of weights (symmetric),
      - or provide `edges` and `edge_weights` as lists.

    The function constructs the Ising matrix A, converts to a QUBO Q,
    and then uses the existing `FEM` class by registering a custom
    expectation function that computes the mean-field expected QUBO value
    under product distributions `p` (p[...,1] = E[x_i]). It runs the
    FEM solver and returns a discrete label array in {0,1} of length n.

    Args:
      coarse_adj_matrix: optional dense adjacency weight matrix (n,n)
      edges, edge_weights: optional edge list and weights (used if coarse_adj_matrix is None)
      c: node weight array length n (int or float)
      k: must be 2 for this implementation
      lambda_penalty: balance penalty multiplier
      num_trials, num_steps, dev: FEM solver settings

    Returns:
      assignment: numpy array shape (n,) with values in {0,1}
    """
    if k != 2:
        raise NotImplementedError("Only k=2 initial partition supported in this helper")

    # Build edges/weights from adjacency matrix if necessary
    if coarse_adj_matrix is not None:
        W = np.array(coarse_adj_matrix, dtype=float)
        assert W.shape[0] == W.shape[1]
        n = W.shape[0]
        edges = []
        weights = []
        # collect upper triangle
        for i in range(n):
            for j in range(i+1, n):
                w = float(W[i, j])
                if w != 0.0:
                    edges.append((i, j))
                    weights.append(w)
    else:
        assert edges is not None and edge_weights is not None
        edges = list(edges)
        weights = list(edge_weights)
        # infer n
        n = int(np.max(np.array(edges)) + 1)

    c = np.asarray(c, dtype=float).reshape(-1)
    assert c.shape[0] == n

    # 1) Build Ising A
    A = build_ising_A(n, edges, weights, c, lambda_penalty)

    # 2) Build QUBO matrix Q
    Q = build_qubo_from_A(A)

    # 3) Wrap into FEM as a customized expected energy using mean-field approximation
    from FEM import FEM as _FEM

    def expected_qubo(_, p: torch.Tensor) -> torch.Tensor:
        # Accept p in multiple shapes and handle flattened inputs produced
        # by OptimizationProblem.inference_value which may vstack per-trial
        # marginals into shape (batch * n, q) or (batch * n,) etc.
        if p.dim() == 3:
            p1 = p[..., 1]
        elif p.dim() == 2:
            # could be (batch, n) or flattened (batch * n, 2)
            if p.shape[1] == n:
                p1 = p
            elif p.shape[1] == 2 and (p.shape[0] % n) == 0:
                batch = p.shape[0] // n
                p_resh = p.reshape(batch, n, 2)
                p1 = p_resh[..., 1]
            else:
                raise ValueError(f"Unexpected 2D p tensor shape: {p.shape}, n={n}")
        elif p.dim() == 1:
            p1 = p.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected p tensor shape: {p.shape}")

        # compute quadratic form p1^T Q p1 using torch
        Q_t = torch.tensor(Q, dtype=p1.dtype, device=p1.device)
        # p1: (batch, n) -> energy per batch
        left = p1 @ Q_t
        vals = (left * p1).sum(dim=1)
        return vals

    def inference_qubo(_, p: torch.Tensor):
        # Convert marginals p into a discrete one-hot config compatible with FEM.
        # Support inputs that may be flattened: (batch * n, q) or (batch * n,)
        if p.dim() == 3:
            # (batch, n, 2) -> argmax over last dim
            idx = p.argmax(dim=2, keepdim=True)
            config = torch.zeros_like(p)
            config.scatter_(2, idx, 1.0)
        elif p.dim() == 2:
            # Could be (batch, n) with probabilities, or flattened (batch*n, 2)
            if p.shape[1] == n:
                p1 = p
                batch, nloc = p1.shape
                config = torch.zeros((batch, nloc, 2), dtype=p1.dtype, device=p1.device)
                x = (p1 >= 0.5).long()
                config[..., 1] = x.float()
                config[..., 0] = 1.0 - config[..., 1]
            elif p.shape[1] == 2 and (p.shape[0] % n) == 0:
                batch = p.shape[0] // n
                p_resh = p.reshape(batch, n, 2)
                idx = p_resh.argmax(dim=2, keepdim=True)
                config = torch.zeros_like(p_resh)
                config.scatter_(2, idx, 1.0)
            else:
                raise ValueError(f"Unexpected 2D p tensor shape for inference: {p.shape}, n={n}")
        elif p.dim() == 1:
            # (n,) -> single sample
            p1 = p.unsqueeze(0)
            batch, nloc = p1.shape
            config = torch.zeros((batch, nloc, 2), dtype=p1.dtype, device=p1.device)
            x = (p1 >= 0.5).long()
            config[..., 1] = x.float()
            config[..., 0] = 1.0 - config[..., 1]
        else:
            raise ValueError(f"Unexpected p tensor shape for inference: {p.shape}")

        return config, torch.zeros(config.shape[0], device=config.device)

    # Build a dummy matrix for FEM API (not used)
    dummy = torch.zeros((n, n), dtype=torch.float32)

    fem = _FEM()
    fem.set_up_problem(n, 0, 'customize', dummy, q=2, customize_expected_func=expected_qubo, customize_infer_func=inference_qubo)
    fem.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=2, manual_grad=False)

    config, result = fem.solve()
    # pick best config
    opt_idx = int(torch.argmin(result).item())
    best = config[opt_idx]
    assignment = best.argmax(dim=1).cpu().numpy().astype(int)
    return assignment


def fem_initial_partition_kway(coarse_adj_matrix: Optional[np.ndarray], edges: Optional[Sequence[Tuple[int,int]]], edge_weights: Optional[Sequence[float]], c: np.ndarray, k: int = 2, lambda_penalty: float = 1.0, num_trials: int = 8, num_steps: int = 200, dev: str = 'cpu') -> np.ndarray:
    """Run FEM directly in q-way bmincut mode (use FEM's bmincut implementation).

    This avoids recursive bisection: it constructs a FEM problem with
    `problem_type='bmincut'` and `q=k`, runs the solver and returns the
    discrete assignment of length n with labels in 0..k-1.
    """
    if k == 2:
        return fem_initial_partition(coarse_adj_matrix, edges, edge_weights, c, k=2, lambda_penalty=lambda_penalty, num_trials=num_trials, num_steps=num_steps, dev=dev)

    # Build adjacency as torch tensor (dense) if needed
    if coarse_adj_matrix is not None:
        W = np.array(coarse_adj_matrix, dtype=float)
        assert W.shape[0] == W.shape[1]
        n = W.shape[0]
        # convert to torch dense
        coupling = torch.tensor(W, dtype=torch.float32)
    else:
        assert edges is not None and edge_weights is not None
        edges = list(edges)
        weights = list(edge_weights)
        n = int(np.max(np.array(edges)) + 1)
        W = np.zeros((n, n), dtype=float)
        for (i, j), w in zip(edges, weights):
            if i != j:
                W[i, j] += w
                W[j, i] += w
        coupling = torch.tensor(W, dtype=torch.float32)

    c = np.asarray(c, dtype=float).reshape(-1)
    assert c.shape[0] == n

    # convert to sparse if large
    try:
        coupling_sparse = coupling.to_sparse()
    except Exception:
        coupling_sparse = coupling

    from FEM import FEM as _FEM

    num_interactions = int((coupling != 0).sum().item() // 2)
    fem = _FEM.from_couplings('bmincut', n, num_interactions, coupling_sparse, node_weights=torch.tensor(c, dtype=torch.float32))
    fem.set_up_solver(num_trials, num_steps, dev=dev, q=k)
    configs, results = fem.solve()
    best_idx = int(torch.argmin(results).item())
    best = configs[best_idx]
    assignment = best.argmax(dim=1).cpu().numpy().astype(int)
    return assignment


 
