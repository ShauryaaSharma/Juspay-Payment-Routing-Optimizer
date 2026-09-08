"""Routing strategies, ordered roughly by sophistication."""

from .base import DiscountedCounts, Router
from .contextual import ContextualThompsonRouter
from .static_weighted import StaticWeightedRouter
from .epsilon_greedy import EpsilonGreedyRouter
from .ucb1 import UCB1Router
from .thompson import ThompsonRouter
from .pid_thompson import PIDThompsonRouter

__all__ = [
    "ContextualThompsonRouter",
    "DiscountedCounts",
    "Router",
    "StaticWeightedRouter",
    "EpsilonGreedyRouter",
    "UCB1Router",
    "ThompsonRouter",
    "PIDThompsonRouter",
]
