import numpy as np

from FEM.cyclic_expansion import cyclic_expansion_refine


def compute_cut(adj, parts):
    n = len(parts)
    cut = 0.0
    for i in range(n):
        for j, w in adj[i]:
            if i < j and parts[i] != parts[j]:
                cut += w
    return cut


def main():
    adj = [[] for _ in range(6)]
    edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 3, 1.0),
        (2, 3, 0.5),
    ]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    partition = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    q = 2

    print('Initial partition:', partition)
    print('Initial cut:', compute_cut(adj, partition))

    new_part = cyclic_expansion_refine(
        adjacency=adj,
        partition=partition,
        q=q,
        max_iterations=10,
        max_candidates=10,
        num_trials=4,
        num_steps=100,
        dev='cpu',
        patience=3,
        verbose=True,
    )

    print('Refined partition:', new_part)
    print('Refined cut:', compute_cut(adj, new_part))


if __name__ == '__main__':
    main()
