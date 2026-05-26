"""Shared helper utilities for hypergraph partition test scripts."""

from __future__ import annotations

import torch


def build_coarse_hyperedges(hyperedges_list, original_to_coarse_map, node_count):
    coarse_hyperedges_list = []
    for he in hyperedges_list:
        coarse_he = list(set(int(original_to_coarse_map[v]) for v in he if v < node_count))
        if len(coarse_he) > 1:
            coarse_hyperedges_list.append(coarse_he)
    return coarse_hyperedges_list


def make_q4_pubo_object(hyperedges_list, node_weights_list, cut_func, num_nodes_local, q_local, imbalance_weight=5.0):
    from FEM.problem import weighted_imbalance_penalty

    class _Q4PUBO:
        def __init__(self):
            self.hyperedges = hyperedges_list
            self.node_weights = torch.tensor(node_weights_list, dtype=torch.float32)
            self.imbalance_weight = imbalance_weight

        def expectation(self, _, p):
            self.node_weights = self.node_weights.to(p.device)
            cut_loss = cut_func(None, p, self.hyperedges)
            imb_penalty = weighted_imbalance_penalty(p, self.node_weights.cpu().numpy())
            return cut_loss + self.imbalance_weight * imb_penalty

        def inference(self, _, p):
            q = q_local
            n = num_nodes_local

            if p.dim() == 2:
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
            return config, torch.zeros(config.shape[0], device=p.device)

    return _Q4PUBO()