"""Interactive API-key onboarding ("login") for the AI providers.

Important, honest note baked into the UX: OpenAI, Claude (Anthropic) and Kimi
(Moonshot) do NOT offer a public OAuth flow that lets a third-party app use your
*consumer subscription* (ChatGPT Plus / Claude Pro / Kimi) for API inference.
Those are separate products; the sanctioned, stable way to use them from any tool
is an **API key** (each has its own free credits / free tier). Reverse-engineering
the web logins violates their ToS and risks account bans, so we don't do that.

This wizard is the legitimate equivalent of "log in with your accounts": it opens
each provider's key page, you paste the key once, and it's stored in your global
config (`~/.config/ofo/.env`, chmod 600). Gemini additionally has a real Google
OAuth path (Cloud/Vertex) for advanced users — noted below.
"""
from __future__ import annotations

import getpass
import os
import sys
import webbrowser

from .config import PROVIDER_KEY_ENVS, config_home

# provider -> (label, key-creation URL, key hint, honest note)
PROVIDERS = {
    "openrouter": ("OpenRouter (gateway, many :free models)",
                   "https://openrouter.ai/keys", "sk-or-...", ""),
    "openai": ("OpenAI (GPT)", "https://platform.openai.com/api-keys", "sk-...",
               "API key — a separate product from ChatGPT Plus. There is no OAuth "
               "to use a ChatGPT subscription via the API."),
    "anthropic": ("Claude (Anthropic)", "https://console.anthropic.com/settings/keys",
                  "sk-ant-...",
                  "API key — separate from Claude Pro. No public OAuth for using a "
                  "subscription via the API."),
    "gemini": ("Google Gemini", "https://aistudio.google.com/apikey", "AIza...",
               "Free tier via an AI Studio key. (Advanced: Google OAuth / Vertex AI "
               "is a real but more involved alternative.)"),
    "moonshot": ("Kimi (Moonshot)", "https://platform.moonshot.ai/console/api-keys",
                 "sk-...",
                 "API key — separate from the Kimi app. No public OAuth for using a "
                 "subscription via the API. (China accounts: set "
                 "MOONSHOT_BASE_URL=https://api.moonshot.cn/v1)"),
    "perplexity": ("Perplexity (Sonar)", "https://www.perplexity.ai/settings/api",
                   "pplx-...", ""),
    "huggingface": ("Hugging Face", "https://huggingface.co/settings/tokens",
                    "hf_...", ""),
    "together": ("Together AI", "https://api.together.ai/settings/api-keys", "", ""),
    "fireworks": ("Fireworks AI", "https://fireworks.ai/account/api-keys", "", ""),
    "replicate": ("Replicate", "https://replicate.com/account/api-tokens", "r8_...", ""),
}

# The four the user usually wants + the OpenRouter gateway.
CORE = ["openrouter", "openai", "anthropic", "gemini", "moonshot"]

ALIASES = {
    "claude": "anthropic", "kimi": "moonshot", "google": "gemini",
    "gpt": "openai", "chatgpt": "openai", "moonshot-ai": "moonshot",
    "hf": "huggingface", "or": "openrouter", "sonar": "perplexity",
}


def env_path() -> str:
    return os.path.join(config_home(), ".env")


def _read_lines(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def update_env(updates: dict) -> str:
    """Set/replace the given KEY=value pairs in the global .env, keeping the rest."""
    path = env_path()
    lines = _read_lines(path)
    done = set()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append("%s=%s" % (key, updates[key]))
                done.add(key)
                continue
        out.append(line)
    for key, val in updates.items():
        if key not in done:
            out.append("%s=%s" % (key, val))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def remove_env(keys) -> str:
    path = env_path()
    drop = set(keys)
    out = []
    for line in _read_lines(path):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            if stripped.split("=", 1)[0].strip() in drop:
                continue
        out.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def resolve(names) -> list:
    out = []
    for raw in names:
        name = ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name in PROVIDERS and name not in out:
            out.append(name)
    return out


def _has_key(provider: str) -> bool:
    return bool(os.environ.get(PROVIDER_KEY_ENVS.get(provider, ""), "").strip())


def run_wizard(ui, providers=None, open_browser: bool = True) -> int:
    selection = providers or CORE
    if not sys.stdin.isatty():
        ui.error("The login wizard needs an interactive terminal.")
        ui.note("Or set keys directly, e.g.:  export OPENAI_API_KEY=sk-...")
        return 2

    ui.note("Provider login — keys are stored in %s (chmod 600)." % env_path())
    ui.warn("These providers use API KEYS, not subscription OAuth — OpenAI/Claude/"
            "Kimi have no public OAuth to use ChatGPT Plus / Claude Pro / Kimi via "
            "the API. Each has its own free credits / free tier.")
    updates = {}
    for provider in selection:
        label, url, hint, note = PROVIDERS[provider]
        env = PROVIDER_KEY_ENVS[provider]
        ui.plain("")
        ui.note("%s  [%s]" % (label, "already set ✓" if _has_key(provider) else "not set"))
        if note:
            ui.plain("    " + note)
        ui.plain("    get a key:  " + url + ("   (%s)" % hint if hint else ""))
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            key = getpass.getpass("    paste %s (hidden — Enter to skip): " % env).strip()
        except (EOFError, KeyboardInterrupt):
            ui.plain("")
            break
        if key:
            updates[env] = key
            ui.note("  ✓ stored " + env)

    if updates:
        path = update_env(updates)
        ui.note("Saved %d key(s) to %s." % (len(updates), path))
        ui.note("They take effect on the next run — try:  ofo --list-models")
    else:
        ui.note("No keys entered — nothing changed.")
    return 0


def handle_login(args, ui) -> int:
    logout = getattr(args, "logout", None)
    if logout:
        names = resolve(str(logout).split(","))
        if not names:
            ui.error("Unknown provider(s). Known: " + ", ".join(PROVIDERS))
            return 2
        remove_env([PROVIDER_KEY_ENVS[p] for p in names])
        ui.note("Removed key(s) for: " + ", ".join(names))
        return 0

    login = getattr(args, "login", None)
    if isinstance(login, str) and login not in ("core", "", "all"):
        selection = resolve(login.split(","))
        if not selection:
            ui.error("Unknown provider(s). Try: openai, claude, gemini, kimi "
                     "(or: " + ", ".join(PROVIDERS) + ")")
            return 2
    elif login == "all":
        selection = list(PROVIDERS.keys())
    else:
        selection = CORE
    return run_wizard(ui, selection)
