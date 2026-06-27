"""Free-Tier AI Orchestrator.

An autonomous, multi-model terminal CLI over free/free-tier AI providers.

Three engines share one model router (auto-select best healthy model per
role + rotate on failure):
  * Agent     - an interactive, tool-using ReAct loop (the default; reads/writes
                files and runs shell commands to actually accomplish goals).
  * Orchestrator - a batch plan -> ensemble -> critique -> integrate -> verify
                pipeline for pure content generation.
  * GodMode   - a parallel fanout mode that asks diverse models the same prompt
                and optionally synthesizes their answers.

A graphical web UI (``ofo --web``) streams the agent's live activity to the
browser over Server-Sent Events.
"""
__version__ = "1.5.0"

from .agent import Agent  # noqa: F401
from .config import Settings  # noqa: F401
from .godmode import GodMode  # noqa: F401
from .orchestrator import Orchestrator  # noqa: F401

__all__ = ["Agent", "GodMode", "Settings", "Orchestrator", "__version__"]
