"""Partition package wrappers for coarsening experiments.

This package exposes lightweight coarsening functions used by tests.
"""
from .coarsen import coarsen_kahypar_like, coarsen_fem_refine_kahypar, evaluate_coarse_cut

__all__ = ["coarsen_kahypar_like", "coarsen_fem_refine_kahypar", "evaluate_coarse_cut"]
