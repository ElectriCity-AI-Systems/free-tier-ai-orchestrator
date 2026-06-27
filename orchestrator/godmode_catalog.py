"""GodMode provider catalogue mapped to this CLI's provider adapters."""
from __future__ import annotations

GODMODE_PROVIDERS = [
    ("ChatGPT", "OpenAI ChatGPT", "openai", "OPENAI_API_KEY",
     "api", "OpenAI API access; ChatGPT-only web features are not automated."),
    ("Claude1", "Anthropic Claude console", "anthropic", "ANTHROPIC_API_KEY",
     "api", "Native Anthropic Messages API."),
    ("Claude2", "Anthropic Claude web", "anthropic", "ANTHROPIC_API_KEY",
     "api", "Native Anthropic Messages API replaces the webview."),
    ("Bard", "Google Bard/Gemini web", "gemini", "GEMINI_API_KEY",
     "api", "Gemini API access; old Bard web login is not automated."),
    ("Perplexity", "Perplexity", "perplexity", "PERPLEXITY_API_KEY",
     "api", "Sonar chat API."),
    ("Perplexity-Labs", "Perplexity Labs", "perplexity", "PERPLEXITY_API_KEY",
     "api", "Use Sonar models or PERPLEXITY_MODELS overrides."),
    ("OpenRouter", "OpenRouter Playground", "openrouter", "OPENROUTER_API_KEY",
     "api", "OpenRouter free/free-tier catalogue."),
    ("Together", "Together Playground", "together", "TOGETHER_API_KEY",
     "api", "Together OpenAI-compatible API."),
    ("HuggingChat", "HuggingChat", "huggingface", "HUGGINGFACE_API_KEY",
     "api", "Hugging Face router/inference providers."),
    ("Oobabooga", "Local text-generation-webui", "oobabooga",
     "optional", "local", "Enable with --providers oobabooga; default base is local."),
    ("Bing", "Microsoft Bing/Copilot web", "-", "-", "webapp-only",
     "No stable public chat API adapter is included."),
    ("Poe", "Quora Poe web", "-", "-", "webapp-only",
     "GodMode uses a webview; this CLI does not automate Poe sessions."),
    ("Phind", "Phind web", "-", "-", "webapp-only",
     "No stable public API adapter is included."),
    ("You.com", "You.com Chat web", "-", "-", "webapp-only",
     "No stable public API adapter is included."),
    ("InflectionPi", "Inflection Pi web", "-", "-", "webapp-only",
     "No stable public API adapter is included."),
    ("StableChat", "Stability AI chat web", "-", "-", "webapp-only",
     "No stable public API adapter is included."),
    ("Falcon180BSpace", "HF Space demo", "-", "-", "webapp-only",
     "Use Hugging Face API-backed models instead."),
    ("Llama2-Lepton", "Lepton Llama demo", "-", "-", "webapp-only",
     "No maintained API adapter is included."),
    ("Vercel", "Vercel AI Chatbot demo", "-", "-", "webapp-only",
     "Demo webapp; no model account adapter."),
    ("Smol", "Smol Talk demo", "-", "-", "webapp-only",
     "Demo webapp; no model account adapter."),
]


def format_godmode_catalog() -> str:
    rows = ["GodMode provider access:"]
    for short, _full, provider, env, status, notes in GODMODE_PROVIDERS:
        target = provider if provider != "-" else "no CLI adapter"
        rows.append("- %s -> %s (%s)" % (short, target, status))
        if env != "-":
            rows.append("  setup: " + env)
        rows.append("  note : " + notes)
    return "\n".join(rows)
