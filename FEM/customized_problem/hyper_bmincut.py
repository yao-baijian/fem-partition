import torch
import torch.nn.functional as Func
import numpy as np

def balance_constrain(J, p, U_max, L_min):
    S_k = p.sum(dim=1)  # [batch, n_clusters] - 完全可微

    # with torch.no_grad():
    #     probabilities = torch.softmax(p, dim=2)
    #     assignments = torch.argmax(probabilities, dim=2)
    #     actual_counts = torch.nn.functional.one_hot(assignments, n_clusters).sum(dim=1)
    #     print(f"actual partition: {actual_counts.tolist()[0]} soft partition: {S_k.tolist()[0]}")
    
    upper_violation = torch.relu(S_k - U_max)
    lower_violation = torch.relu(L_min - S_k)

    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    return balance_loss

def balance_constrain_softplus(J, p, U_max, L_min):
    # softplus 
    # S_k = p.sum(dim=1)
    # softplus(x) = log(1 + exp(beta * x)) / beta
    # upper_violation = torch.log(1 + torch.exp(beta * (S_k - U_max))) / beta
    # lower_violation = torch.log(1 + torch.exp(beta * (L_min - S_k))) / beta

    # upper_violation = torch.nn.functional.softplus(beta *(S_k - U_max))
    # lower_violation = torch.nn.functional.softplus(beta *(L_min - S_k))

    # upper_violation = torch.where(
    #     x > 20,
    #     x / beta,
    #     torch.log1p(torch.exp(x)) / beta 
    # )

    # balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    # print(f"balance_loss: {balance_loss}")
    # return balance_loss
    pass

def balance_constrain_relu(J, p, U_max, L_min):
    # relu hard regu
    S_k = p.sum(dim=1)
    config, result = infer_hyperbmincut(J, p)
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    best_config = config[optimal_inds[0]]
    group_assignment = best_config.argmax(dim=1).cpu().numpy()
    group_counts = np.bincount(group_assignment, minlength=4)

    batch_size, n_nodes, n_clusters = p.shape
    
    probabilities = torch.softmax(p, dim=2)
    assignments = torch.argmax(probabilities, dim=2)
    one_hot = torch.nn.functional.one_hot(assignments, num_classes=n_clusters)
    S_k = one_hot.sum(dim=1).float()

    print(f"group_counts: {group_counts}")
    print(f"S_k: {S_k}")
    upper_violation = torch.relu(S_k - U_max)
    lower_violation = torch.relu(L_min - S_k)
    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)

    return balance_loss

def infer_hyperbmincut(J, p, hyperedges):
    config = Func.one_hot(p.view(-1,J.shape[0],p.shape[-1]).argmax(dim=2), num_classes=p.shape[-1]).to(J.dtype)
    return config, expected_hyperbmincut(J, config, hyperedges) / 2

def expected_hyperbmincut(J, p, hyperedges):
    # return ((J @ p) * (1-p)).sum((1, 2))
    # n_hyperedges = J.shape
    # n_groups = p.shape[1]
    
    # log_prob_no_node = torch.zeros(n_hyperedges, n_groups, device=p.device)
    
    # for e in range(n_hyperedges):
    #     nodes_in_e = torch.where(J[e] > 0)[0]
    #     if len(nodes_in_e) > 0:
    #         log_one_minus_p = torch.log(1 - p[nodes_in_e] + 1e-10) 
    #         log_prob_no_node[e] = torch.sum(log_one_minus_p, dim=0)
    
    # prob_no_node = torch.exp(log_prob_no_node)
    # p_ek = 1 - prob_no_node
    # expected_lambda = p_ek.sum(dim=1)
    # expected_cut_value = (expected_lambda - 1).sum(1)
    
    # return expected_cut_value

    total_cut_value = 0.0
    
    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        m = p.shape[2]

        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]

        # 最简单的连续映射: 跨区数 = m - (m-1) * max_ratio
        max_ratio = torch.max(expected_nodes_per_cluster, dim=1)[0] / k
        expected_crossing = m * (1 - max_ratio)
        
        total_cut_value += expected_crossing
    
    # print(f"Weighted Cut value: {total_cut_value}")
    return total_cut_value

    # threshold=15
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     batch_size, _, num_clusters = he_probs.shape
        
    #     if k <= threshold:
    #         # 对于小k，使用精确计算
    #         prob_single_cluster = torch.zeros(batch_size, device=p.device)
    #         for cluster_j in range(num_clusters):
    #             prob_all_in_j = torch.prod(he_probs[:, :, cluster_j], dim=1)
    #             prob_single_cluster += prob_all_in_j
            
    #         cut_expectation = 1 - torch.clamp(prob_single_cluster, 0.0, 1.0)
            
    #     else:
    #         # 对于大k，使用成对近似
    #         # 计算节点分配的"集中度"
    #         cluster_weights = torch.sum(he_probs, dim=1) / k  # [batch, num_clusters]
    #         max_concentration = torch.max(cluster_weights, dim=1)[0]  # 最大簇的节点比例
            
    #         # 集中度越高，越不可能被切割
    #         cut_expectation = 1 - max_concentration
        
    #     cut_value = cut_expectation * weight
    #     total_cut_value = total_cut_value + cut_value
        
    #     # if he_idx < 3:
    #     #     method = "精确" if k <= threshold else "近似"
    #     #     print(f"超边{he_idx}(k={k}, {method}): cut期望={cut_expectation[0]:.6f}")
    
    # print(f"Hybrid Cut value: {total_cut_value}")
    # return total_cut_value


    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     expected_nodes = torch.sum(he_probs, dim=1)  # [batch, m]
        
    #     # 更好的近似：考虑概率的方差
    #     p_used = torch.zeros_like(expected_nodes)
        
    #     for cluster_j in range(he_probs.shape[2]):
    #         probs_j = he_probs[:, :, cluster_j]  # [batch, k]
    #         e_j = expected_nodes[:, cluster_j]   # 期望节点数
            
    #         # 使用一阶泰勒展开近似
    #         # log(P(无节点)) = ∑ log(1-p_i,j) ≈ -∑ p_i,j - 0.5∑ p_i,j^2
    #         sum_p = torch.sum(probs_j, dim=1)
    #         sum_p2 = torch.sum(probs_j**2, dim=1)
            
    #         # P(无节点) ≈ exp(-sum_p - 0.5*sum_p2)
    #         p_no_nodes = torch.exp(-sum_p - 0.5 * sum_p2)
    #         p_used[:, cluster_j] = 1 - p_no_nodes
        
    #     expected_crossing = torch.sum(p_used, dim=1)
    #     cut_value = torch.relu(expected_crossing - 1) * weight
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value
    
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     batch_size, _, num_clusters = he_probs.shape
        
    #     # 使用log空间计算避免数值下溢
    #     log_prob_single_cluster = None
        
    #     for cluster_j in range(num_clusters):
    #         # log(P(所有节点在簇j)) = ∑ log(p_i,j)
    #         log_probs = torch.log(he_probs[:, :, cluster_j] + 1e-12)
    #         log_prob_all_in_j = torch.sum(log_probs, dim=1)
            
    #         if log_prob_single_cluster is None:
    #             log_prob_single_cluster = log_prob_all_in_j
    #         else:
    #             # log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    #             max_log = torch.max(log_prob_single_cluster, log_prob_all_in_j)
    #             log_prob_single_cluster = max_log + torch.log(
    #                 torch.exp(log_prob_single_cluster - max_log) + 
    #                 torch.exp(log_prob_all_in_j - max_log) + 1e-12
    #             )
        
    #     prob_single_cluster = torch.exp(log_prob_single_cluster)
        
    #     # 确保概率在合理范围内
    #     prob_single_cluster = torch.clamp(prob_single_cluster, 0.0, 1.0)
    #     cut_expectation = 1 - prob_single_cluster
        
    #     cut_value = cut_expectation * weight
    #     total_cut_value = total_cut_value + cut_value
        
    #     if he_idx < 2:  # 只打印前两个超边
    #         print(f"超边{he_idx}(k={k}): P(单簇)={prob_single_cluster[0]:.8f}")
    
    # print(f"Total Cut value: {total_cut_value}")
    # return total_cut_value
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
    #     m = p.shape[2]
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
        
    #     # 计算一阶矩：E[X_j]
    #     p_used = 1 - torch.prod(1 - he_probs, dim=1)  # [batch, m]
    #     mean_crossing = torch.sum(p_used, dim=1)
        
    #     # 计算二阶矩：E[X_j X_k]
    #     second_moment = 0.0
    #     for j in range(m):
    #         for l in range(j+1, m):
    #             # P(簇j和簇l都被使用) = 1 - P(不用j) - P(不用l) + P(都不用j和l)
    #             p_no_j = torch.prod(1 - he_probs[:, :, j], dim=1)
    #             p_no_l = torch.prod(1 - he_probs[:, :, l], dim=1)
    #             p_no_jl = torch.prod(1 - he_probs[:, :, j] - he_probs[:, :, l], dim=1)
    #             p_both_used = 1 - p_no_j - p_no_l + p_no_jl
    #             second_moment += p_both_used
        
    #     # 使用切比雪夫不等式近似 P(跨区数 ≥ 2)
    #     variance = second_moment + mean_crossing - mean_crossing**2
    #     # P(跨区数 ≥ 2) ≈ 1 - P(跨区数 ≤ 1)
    #     # 但更实用：cut_value = (mean_crossing - 1) 当 mean_crossing > 1.5，否则用概率近似
    #     cut_prob = torch.sigmoid((mean_crossing - 1.5) * 10)  # 平滑的阶跃函数
    #     cut_value = cut_prob * weight
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

def expected_hyperbmincut_expected_nodes_temped(J, p, hyperedges):
    total_cut_value = 0.0
    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        
        he_probs = p[:, he, :]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)
        
        temperature = 0.1
        weights = torch.softmax(expected_nodes_per_cluster / temperature, dim=1)
        weighted_max = torch.sum(weights * expected_nodes_per_cluster, dim=1)
        
        cut_value = 1 - (weighted_max / k)
        total_cut_value = total_cut_value + cut_value * weight
        
    return total_cut_value

def expected_hyperbmincut_max_expected_nodes(J, p, hyperedges):
    total_cut_value = 0.0

    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        
        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]
        max_expected_nodes = torch.max(expected_nodes_per_cluster, dim=1)[0]
        cut_value = 1 - (max_expected_nodes / k)
        total_cut_value = total_cut_value + cut_value

    print(f"Weighted Cut value: {total_cut_value}")
    return total_cut_value

def expected_hyperbmincut_explicit(J, p, hyperedges):
    # Vectorized implementation across hyperedges grouped by edge-size
    device = p.device
    m = 4  # fixed cluster count for this explicit routine

    # Precompute masks on device
    pair_masks = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32, device=device)

    triple_masks = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ], dtype=torch.float32, device=device)

    # Group hyperedges by size to batch-gather p efficiently
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, he in enumerate(hyperedges):
        groups[len(he)].append((idx, he))

    total_cut = None
    for k, he_list in groups.items():
        # build index tensor: (num_edges_in_group, k)
        idxs = torch.tensor([he for (_, he) in he_list], dtype=torch.long, device=device)
        # gather probabilities: p has shape [batch, n, m]
        # result: [batch, num_edges, k, m]
        he_probs = p[:, idxs, :]

        # prod over nodes -> [batch, num_edges, m]
        prod_k = torch.prod(he_probs, dim=2)
        prob_single = prod_k.sum(dim=2)  # [batch, num_edges]

        batch_size = he_probs.shape[0]

        # pairs
        # pair computation: for each pair-mask, sum probabilities over selected clusters per node, then prod over nodes
        he_probs_exp = he_probs.unsqueeze(2)  # [batch, num_edges, 1, k, m]
        pair_masks_expanded = pair_masks.view(1, 1, 6, 1, 4).to(device)  # [1,1,6,1,4]
        sum_clusters = (he_probs_exp * pair_masks_expanded).sum(dim=4)  # [batch, num_edges, 6, k]
        pair_probs = torch.prod(sum_clusters, dim=3)  # [batch, num_edges, 6]
        sum_2comb = pair_probs.sum(dim=2)  # [batch, num_edges]
        prob_2 = sum_2comb - 2 * prob_single

        # triples
        he_probs_exp_t = he_probs.unsqueeze(2)  # [batch, num_edges, 1, k, m]
        triple_masks_expanded = triple_masks.view(1, 1, 4, 1, 4).to(device)
        sum_clusters_t = (he_probs_exp_t * triple_masks_expanded).sum(dim=4)  # [batch, num_edges, 4, k]
        triple_probs = torch.prod(sum_clusters_t, dim=3)  # [batch, num_edges, 4]
        sum_3comb = triple_probs.sum(dim=2)  # [batch, num_edges]
        prob_3 = sum_3comb - 2 * sum_2comb + 3 * prob_single

        prob_4 = 1.0 - prob_single - prob_2 - prob_3

        contrib = prob_2 + 2.0 * prob_3 + 3.0 * prob_4  # [batch, num_edges]
        # accumulate
        if total_cut is None:
            total_cut = contrib.sum(dim=1)
        else:
            total_cut = total_cut + contrib.sum(dim=1)

    if total_cut is None:
        # no hyperedges -> zero
        return torch.zeros(p.shape[0], device=device)
    return total_cut

        # total_cut_value = total_cut_value + prob_2_clusters + prob_3_clusters * 2 + prob_4_clusters * 3
        
        # if he_idx == 0:
        #     print(f"超边{k}: P(1簇)={prob_single_cluster[0]:.6f}, P(2簇)={prob_2_clusters[0]:.6f}, "
        #           f"P(3簇)={prob_3_clusters[0]:.6f}, P(4簇)={prob_4_clusters[0]:.6f}, "
        #           f"Cut期望={cut_expectation[0]:.6f}")
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

def expected_hyperbmincut_expected_crossing_simplified(J, p, hyperedges):
    total_cut_value = 0.0
    
    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        
        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)
        e = expected_nodes_per_cluster / k  # 归一化的期望节点数
        p_used = 1 - torch.exp(-k * e)
        
        # 期望跨区数
        expected_crossing = torch.sum(p_used, dim=1)
        
        # 近似方差（假设独立）
        variance = torch.sum(p_used * (1 - p_used), dim=1)
        
        # 使用正态近似计算 P(跨区数 ≥ 2)
        # 但更简单：cut_value = max(0, expected_crossing - 1)
        cut_value = torch.relu(expected_crossing - 1) * weight
        total_cut_value = total_cut_value + cut_value
    
    print(f"Weighted Cut value: {total_cut_value}")
    return total_cut_value

def manual_grad_hyperbmincut(J, p, U_max, L_min, n, h, imbalance_weight, q):
    
    # 1. 计算概率分布 (需要softmax)
    # p = torch.softmax(h, dim=-1)  # [batch, n_nodes, q]
    
    group_sizes = p.sum(dim=1)  # [batch, q] - 每个分组的"软节点数"
    
    temperature = 0.1
    indicator_upper = torch.sigmoid((group_sizes - U_max) / temperature)  
    indicator_lower = torch.sigmoid((L_min - group_sizes) / temperature)  
    
    balance_grad =  imbalance_weight * (indicator_upper - indicator_lower)  # [batch, q]
    
    # 扩展到每个节点：balance_grad_expanded[i,k] = balance_grad[batch,k]
    balance_grad_expanded = balance_grad.unsqueeze(1).expand(-1, n, -1)  # [batch, n_nodes, q]
    
    cut_grad = torch.zeros_like(h)
    for k in range(q):
        p_k = p[:, :, k]  # [batch, n_nodes]
        for b in range(batch_size):
            cut_grad[b, :, k] = torch.matmul(J, 1 - 2 * p_k[b])
    
    total_grad = cut_grad + balance_grad_expanded
    
    return total_grad

# elif self.problem_type == 'hyperbmincut':
#     # print(f"expected_hyperbmincut: {expected_hyperbmincut(self.coupling_matrix, p)}")
#     # print(f"Balance loss: {self.imbalance_weight * balance_constrain_1(p, self.U_max, self.L_min)}")
#     # factor = (step_max - step )/ step_max
#     # rev_factor = ( step )/ step_max
#     expect_loss = expected_hyperbmincut(self.coupling_matrix, p, self.hyperedge)
#     balance_loss = self.imbalance_weight * balance_constrain(self.coupling_matrix, p, self.U_max, self.L_min)
#     return expect_loss, balance_loss