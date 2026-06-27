"""Configuration, constants and environment loading.

No third-party dependencies. The whole orchestrator runs on the Python
standard library; `rich` is used only for prettier output when present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# Base URLs are overridable for proxies, gateways or self-hosted compatibles.
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
PERPLEXITY_BASE_URL = os.environ.get(
    "PERPLEXITY_BASE_URL", "https://api.perplexity.ai").rstrip("/")
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")
TOGETHER_BASE_URL = os.environ.get(
    "TOGETHER_BASE_URL", "https://api.together.xyz/v1").rstrip("/")
HUGGINGFACE_BASE_URL = os.environ.get(
    "HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
REPLICATE_BASE_URL = os.environ.get(
    "REPLICATE_BASE_URL", "https://api.replicate.com/v1").rstrip("/")
# Moonshot / Kimi (OpenAI-compatible). Use api.moonshot.cn for China accounts.
MOONSHOT_BASE_URL = os.environ.get(
    "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")

# Sent to OpenRouter for attribution / leaderboard ranking. Harmless.
APP_REFERER = "https://github.com/local/openrouter-free-orchestrator"
APP_TITLE = "Free-Tier AI Orchestrator"

API_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDER_KEY_ENVS = {
    "openrouter": API_KEY_ENV,
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "oobabooga": "OOBABOOGA_API_KEY",
}
PROVIDER_ORDER = (
    "openrouter", "openai", "anthropic", "gemini", "perplexity", "moonshot",
    "huggingface", "together", "fireworks", "replicate", "oobabooga",
)
KEYLESS_PROVIDERS = ("oobabooga",)
PROVIDER_ENV = "OFO_PROVIDERS"


def config_home() -> str:
    """Global config dir: $XDG_CONFIG_HOME/ofo (default ~/.config/ofo)."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(xdg, "ofo")


def _load_env_file(path: str) -> None:
    try:
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)
    except OSError:
        pass


def load_dotenv(path: str = ".env") -> None:
    """Dependency-free .env loader that works from any directory.

    Real environment variables always win. Among files, the first to set a key
    wins, so precedence is:
        $OFO_ENV_FILE  >  ./.env  >  ~/.config/ofo/.env  >  ~/.ofo/.env
    This lets a globally-installed ``ofo`` find your keys no matter which folder
    you run it in, while a project-local ``.env`` still overrides the global one.
    """
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("OFO_ENV_FILE", ""),
        path,
        os.path.join(config_home(), ".env"),
        os.path.join(home, ".ofo", ".env"),
    ]
    seen = set()
    for cand in candidates:
        if not cand:
            continue
        absolute = os.path.abspath(os.path.expanduser(cand))
        if absolute in seen:
            continue
        seen.add(absolute)
        _load_env_file(absolute)


@dataclass
class Settings:
    """All tunable knobs for a run. Sensible, *safe* defaults."""

    api_key: str = ""  # OpenRouter key, kept for backwards compatibility.
    provider_api_keys: Dict[str, str] = field(default_factory=dict)
    enabled_providers: str = ""       # comma-separated, empty means all keyed providers
    provider_model_limit: int = 160   # per provider; keeps huge catalogues manageable

    # --- networking / safety rails ---
    max_concurrency: int = 3          # simultaneous in-flight requests
    request_timeout: float = 120.0    # seconds per HTTP request
    max_retries: int = 4              # retries for transient errors (per model)
    min_request_interval: float = 0.25  # gentle pacing between request starts
    model_cooldown_seconds: float = 45.0  # bench a model after a 429/error

    # --- orchestration budgets (hard stops against runaway loops) ---
    max_task_attempts: int = 3        # critic-driven retries per subtask
    acceptance_threshold: int = 80    # critic score (0-100) needed to "pass"
    ensemble_size: int = 3            # models that race on a hard subtask
    enable_ensemble: bool = True
    refine_passes: int = 1            # whole-deliverable refinement rounds
    max_total_requests: int = 200     # global request budget for one goal
    max_wallclock_seconds: float = 1800.0  # global time budget (informational)

    # --- generation ---
    temperature: float = 0.4
    max_tokens: int = 2048

    # --- agent (interactive tool-use) mode ---
    agent_role: str = "reasoning"     # driver role for the agent loop
    max_steps: int = 25               # max tool-actions per goal (loop guard)
    verify_finish: bool = True        # cross-model check before accepting "finish"
    allow_bash: bool = True           # expose the run_bash tool
    allow_outside: bool = False       # allow file access outside the working dir
    auto_approve: bool = False        # skip confirmations for writes/commands
    bash_timeout: float = 60.0        # per-command timeout (seconds)
    max_tool_output: int = 4000       # cap on tool observation size (chars)

    # --- godmode fanout mode ---
    godmode_width: int = 4            # diverse models to ask in parallel
    godmode_synthesize: bool = True   # merge the panel answers by default
    consult_default_models: int = 8   # consult_models default panel size when the
                                      # agent doesn't pass max_models (was: ALL models)

    # --- efficiency & self-learning ---
    use_cache: bool = True            # dedupe identical deterministic calls
    cache_temp_threshold: float = 0.25  # only cache calls at/below this temp
    cache_max_entries: int = 512
    enable_stats: bool = True         # persist a cross-session model leaderboard
    stats_path: str = ""              # resolved to ~/.ofo/leaderboard.json if empty

    # --- io / behaviour ---
    output_path: Optional[str] = None
    assume_yes: bool = False
    no_color: bool = False
    verbose: bool = False

    def redacted_key(self) -> str:
        if not self.api_key:
            return "<missing>"
        k = self.api_key
        return (k[:6] + "…" + k[-4:]) if len(k) > 14 else "<set>"

    def keyed_providers(self):
        keys = dict(self.provider_api_keys or {})
        if self.api_key:
            keys.setdefault("openrouter", self.api_key)
        return [p for p in PROVIDER_ORDER if keys.get(p)]
