"""Investigation agent layer on top of the payment router.

The router detects that something broke; this layer works out what. Entry
point is `Investigator.investigate(alert, window)`.
"""

from .investigator import Investigator, InvestigatorConfig, summarise
from .llm import AnthropicClient, BaselinePolicyClient, default_client
from .loop import AgentLoop, LoopBudget
from .memory import Memory, MemoryStore
from .schemas import Diagnosis, InvestigationResult, Usage
from .telemetry import TelemetryStore, format_clock
from .tools import ToolDispatcher, tool_definitions

__all__ = [
    "AgentLoop",
    "AnthropicClient",
    "BaselinePolicyClient",
    "Diagnosis",
    "InvestigationResult",
    "Investigator",
    "InvestigatorConfig",
    "LoopBudget",
    "Memory",
    "MemoryStore",
    "TelemetryStore",
    "ToolDispatcher",
    "Usage",
    "default_client",
    "format_clock",
    "summarise",
    "tool_definitions",
]
