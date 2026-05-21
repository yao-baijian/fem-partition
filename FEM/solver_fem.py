import torch
from .problem import *
from math import log
from .utils import *

def entropy_q(p):
    """
    p is the probabilities for each group, shape [batch, N, q], with q denoting the number of groups
    return - \sum_{i=1}^N sum_{t=1}^q p(t)*\log p(t)
    """
    return - (p*torch.log(p)).sum(2).sum(1)

def entropy_grad_q(p):
    return -p * (torch.log(p) - (p*torch.log(p)).sum(2,keepdim=True).expand(p.shape))

def entropy_binary(p):
    return - ((p*torch.log(p)) + (1-p)*torch.log(1-p)).sum(1)

def entropy_grad_binary(p):
    grad = - (p * (1-p) * (p.log() - (1-p).log()))
    return grad


class Solver:
    def __init__(
            self, 
            problem, num_trials, num_steps, betamin=0.01, betamax=0.5, 
            anneal='inverse', optimizer='adam', learning_rate=0.1, dev='cpu', 
            dtype=torch.float32, seed=1, q=2, manual_grad=False, 
            h_factor=0.01, sparse=False, drawer = None
        ):
        self.dtype = dtype
        self.dev = dev
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax),num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev) 
        self.num_trials = num_trials
        self.seed = seed
        self.q = q
        self.manual_grad = manual_grad
        self.h_factor = h_factor
        self.problem = problem
        self.problem.set_up_couplings_status(dev, dtype)
        self.problem.extra_preparation(num_trials, sparse)
        self.binary = True if self.problem.problem_type in ['maxcut', 'vertexcover'] else False
        if self.binary:
            assert self.q == 2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.drawer = drawer

    def initialize(self):
        torch.manual_seed(self.seed)
        if self.binary:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.problem.num_nodes], 
                device=self.dev, dtype=self.dtype
            )
        else:
# **********************************   Start   *********************************** #
            h = self.h_factor * torch.randn(
                [self.num_trials, self.problem.num_nodes, self.q], 
                device=self.dev, dtype=self.dtype
            )

            # h = self.h_factor * torch.randn(
            #     [self.num_trials, self.problem.num_nodes, self.q], 
            #     device=self.dev, dtype=self.dtype
            # )

            # # 改进的多分类初始化
            # n_trials = self.num_trials
            # n_nodes = self.problem.num_nodes
            # q = self.q
            
            # # 方法1：为每个节点随机选择一个主簇并增强
            # h = torch.randn(n_trials, n_nodes, q, device=self.dev, dtype=self.dtype) * 0.2
            
            # # 为每个节点随机选择主簇
            # main_clusters = torch.randint(0, q, (n_trials, n_nodes), device=self.dev)
            
            # # 使用scatter_高效地增强主簇
            # batch_indices = torch.arange(n_trials, device=self.dev).unsqueeze(1).expand(-1, n_nodes)
            # node_indices = torch.arange(n_nodes, device=self.dev).unsqueeze(0).expand(n_trials, -1)
            
            # h[batch_indices, node_indices, main_clusters] += 3.0
            
            # # 乘以h_factor保持原有缩放
            # h = self.h_factor * h

        if self.manual_grad:
            h.requires_grad=False
        else:
            h.requires_grad=True
        return h
    
    def set_up_optimizer(self, params):
        if self.optimizer == 'adam':
            self.opt = torch.optim.Adam([params], lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.opt = torch.optim.RMSprop(
                [params], lr=self.learning_rate, alpha=0.98, eps=1e-08, 
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unkown optimizer, valid choices are ['adam', 'rmsprop'].")

    def set_up_optimizer_placement(self, params):
        if self.optimizer == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.opt = torch.optim.RMSprop(
                params, lr=self.learning_rate, alpha=0.98, eps=1e-08, 
                weight_decay=0.01, momentum=0.91, centered=False
            )
    
    def iterate(self):
        h = self.initialize()
        self.set_up_optimizer(h)
        step_max = len(self.betas)
        for step in range(step_max):
            p = torch.sigmoid(h) if self.binary else torch.softmax(h, dim=2)
            self.opt.zero_grad()
            if self.binary:
                entropy_grad = entropy_grad_binary
                entropy = entropy_binary
            else:
                entropy_grad = entropy_grad_q
                entropy = entropy_q
            if self.manual_grad:
                h.grad = self.problem.manual_grad(p) - \
                    entropy_grad(p) / self.betas[step]
            else:
                free_energy = self.problem.expectation(p) \
                    - entropy(p) / self.betas[step]
                free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy
            self.opt.step()

        # self.drawer.draw_multi_step_placement()

        return p

    def solve(self):
        marginal = self.iterate()
        configs, results = self.problem.inference_value(marginal)
        return configs, results