"""Command-line interface.

Default mode is an interactive, tool-using agent (a Claude-Code-style REPL).
Pass a goal as an argument to run the agent once and exit. Use --pipeline for
the batch plan/ensemble/verify engine instead.

    ofo                          # interactive agent shell
    ofo "add a /health route"    # run the agent once on this goal
    ofo --pipeline "write a spec" # batch multi-model pipeline
    ofo --list-models            # show discovered free/free-tier models
    ofo --self-test              # offline tests, no key needed
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .agent import Agent
from .client import AuthError, BudgetExceeded, OpenRouterClient, OpenRouterError
from .config import (API_KEY_ENV, KEYLESS_PROVIDERS, PROVIDER_ENV,
                     PROVIDER_KEY_ENVS, Settings, load_dotenv)
from .godmode_catalog import format_godmode_catalog
from .godmode import GodMode, format_report
from .orchestrator import Orchestrator
from .registry import (ROLE_CODER, ROLE_GENERAL, ROLE_REASONING, ModelRegistry)
from .repl import run_repl
from .router import ModelRouter
from .tools import ToolBox
from .ui import UI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ofo",
        description="Autonomous, tool-using agent powered by free/free-tier AI models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set one or more provider keys in your environment or a .env file "
               "(OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
               "PERPLEXITY_API_KEY, GEMINI_API_KEY, HUGGINGFACE_API_KEY, "
               "TOGETHER_API_KEY, FIREWORKS_API_KEY, REPLICATE_API_TOKEN; "
               "local Oobabooga uses --providers oobabooga).",
    )
    p.add_argument("goal", nargs="*", help="Goal to run once; omit for an interactive shell.")
    p.add_argument("--version", action="version", version="ofo " + __version__)
    p.add_argument("--web", action="store_true",
                   help="Launch the graphical web UI in your browser.")
    p.add_argument("--web-port", type=int, default=None,
                   help="Port for the web UI (default 8765).")
    p.add_argument("--list-models", action="store_true",
                   help="List discovered free/free-tier models and exit.")
    p.add_argument("--stats", action="store_true",
                   help="Show the learned model leaderboard and exit.")
    p.add_argument("--self-test", action="store_true",
                   help="Run offline self-tests (no key/network) and exit.")
    p.add_argument("--godmode-providers", action="store_true",
                   help="Show how smol-ai/GodMode providers map to CLI access.")

    a = p.add_argument_group("agent mode (default)")
    a.add_argument("--root", default=None,
                   help="Working directory the agent operates in (default: cwd).")
    a.add_argument("--auto", action="store_true",
                   help="Auto-approve file writes & shell commands (still blocks "
                        "obviously destructive commands).")
    a.add_argument("--no-bash", action="store_true",
                   help="Disable the run_bash tool entirely.")
    a.add_argument("--allow-outside", action="store_true",
                   help="Allow file access outside the working directory.")
    a.add_argument("--max-steps", type=int, default=None,
                   help="Max tool-actions per goal (default 25).")
    a.add_argument("--no-verify", action="store_true",
                   help="Don't run the cross-model check before finishing.")
    a.add_argument("--role", choices=[ROLE_REASONING, ROLE_CODER, ROLE_GENERAL],
                   default=None, help="Driver role for the agent (default reasoning).")

    g = p.add_argument_group("godmode mode")
    g.add_argument("--godmode", action="store_true",
                   help="Ask several diverse models in parallel, show all answers, "
                        "and synthesize them.")
    g.add_argument("--godmode-width", type=int, default=None,
                   help="GodMode: number of diverse models to ask (default 4).")
    g.add_argument("--godmode-no-synthesis", action="store_true",
                   help="GodMode: show the parallel answers without merging them.")
    g.add_argument("--consult-cap", type=int, default=None,
                   help="Default panel size for the agent's consult_models tool "
                        "(default 8). The agent can still pass max_models to widen.")

    b = p.add_argument_group("pipeline mode (--pipeline)")
    b.add_argument("--pipeline", action="store_true",
                   help="Use the batch plan/ensemble/verify engine instead of the agent.")
    b.add_argument("--dry-run", action="store_true",
                   help="Pipeline: print the plan only, do not execute.")
    b.add_argument("--max-attempts", type=int, default=None,
                   help="Pipeline: critic retries per subtask (default 3).")
    b.add_argument("--threshold", type=int, default=None,
                   help="Acceptance score 0-100 (default 80).")
    b.add_argument("--ensemble", type=int, default=None,
                   help="Pipeline: models racing on hard subtasks (default 3).")
    b.add_argument("--no-ensemble", action="store_true", help="Disable ensembling.")
    b.add_argument("--refine", type=int, default=None,
                   help="Pipeline: deliverable refinement passes (default 1).")

    s = p.add_argument_group("safety / budgets")
    s.add_argument("--max-requests", type=int, default=None,
                   help="Hard cap on total API requests (default 200).")
    s.add_argument("--concurrency", type=int, default=None,
                   help="Max simultaneous requests (default 3).")
    s.add_argument("--timeout", type=float, default=None,
                   help="Per-request timeout in seconds (default 120).")
    s.add_argument("--no-cache", action="store_true",
                   help="Disable the deterministic response cache.")
    s.add_argument("--no-stats", action="store_true",
                   help="Don't read/write the cross-session model leaderboard.")
    s.add_argument("--providers", default=None,
                   help="Comma-separated provider allow-list (default: all keyed providers).")

    o = p.add_argument_group("output")
    o.add_argument("-o", "--output", default=None, help="Write the final result to a file.")
    o.add_argument("-y", "--yes", action="store_true", help="Assume 'yes' for file writes.")
    o.add_argument("--no-color", action="store_true", help="Disable colour/rich output.")
    o.add_argument("--max-tokens", type=int, default=None, help="Max tokens/generation (2048).")
    o.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0.4).")
    o.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def _settings_from_args(args) -> Settings:
    s = Settings()
    s.api_key = os.environ.get(API_KEY_ENV, "").strip()
    s.provider_api_keys = {
        provider: os.environ.get(env_name, "").strip()
        for provider, env_name in PROVIDER_KEY_ENVS.items()
        if os.environ.get(env_name, "").strip()
    }
    if s.api_key:
        s.provider_api_keys["openrouter"] = s.api_key
    if args.providers is not None:
        s.enabled_providers = args.providers
    else:
        s.enabled_providers = os.environ.get(PROVIDER_ENV, "")
    # agent
    if args.max_steps is not None:
        s.max_steps = max(1, args.max_steps)
    if args.no_verify:
        s.verify_finish = False
    if args.no_bash:
        s.allow_bash = False
    if args.allow_outside:
        s.allow_outside = True
    if args.auto:
        s.auto_approve = True
    if args.role is not None:
        s.agent_role = args.role
    # godmode
    if args.godmode_width is not None:
        s.godmode_width = max(1, args.godmode_width)
    if args.godmode_no_synthesis:
        s.godmode_synthesize = False
    if args.consult_cap is not None:
        s.consult_default_models = max(1, args.consult_cap)
    # pipeline
    if args.max_attempts is not None:
        s.max_task_attempts = max(1, args.max_attempts)
    if args.threshold is not None:
        s.acceptance_threshold = max(0, min(100, args.threshold))
    if args.no_ensemble:
        s.enable_ensemble = False
    if args.ensemble is not None:
        s.ensemble_size = max(1, args.ensemble)
        s.enable_ensemble = args.ensemble > 1
    if args.refine is not None:
        s.refine_passes = max(0, args.refine)
    # safety / efficiency
    if args.max_requests is not None:
        s.max_total_requests = max(1, args.max_requests)
    if args.concurrency is not None:
        s.max_concurrency = max(1, args.concurrency)
    if args.timeout is not None:
        s.request_timeout = max(5.0, args.timeout)
    if args.no_cache:
        s.use_cache = False
    if args.no_stats:
        s.enable_stats = False
    # output
    if args.max_tokens is not None:
        s.max_tokens = max(256, args.max_tokens)
    if args.temperature is not None:
        s.temperature = max(0.0, min(2.0, args.temperature))
    s.output_path = args.output
    s.assume_yes = args.yes
    s.no_color = args.no_color
    s.verbose = args.verbose
    return s


def _stdin_goal() -> str:
    return "" if sys.stdin.isatty() else sys.stdin.read().strip()


def _allowed_keyless_only(providers: str) -> bool:
    raw = providers or os.environ.get(PROVIDER_ENV, "")
    selected = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return bool(selected) and selected.issubset(set(KEYLESS_PROVIDERS))


def _require_key(ui: UI, providers: str = "") -> bool:
    if any(os.environ.get(env_name, "").strip()
           for env_name in PROVIDER_KEY_ENVS.values()):
        return True
    if _allowed_keyless_only(providers):
        return True
    ui.error("No API key found.")
    ui.note("Set at least one provider key in your environment or .env:")
    ui.note("  OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,")
    ui.note("  PERPLEXITY_API_KEY, GEMINI_API_KEY, HUGGINGFACE_API_KEY,")
    ui.note("  TOGETHER_API_KEY, FIREWORKS_API_KEY, REPLICATE_API_TOKEN")
    ui.note("For local Oobabooga only: use --providers oobabooga.")
    return False


def _maybe_write(ui: UI, settings: Settings, text: str) -> None:
    if not settings.output_path:
        return
    if ui.confirm("Write result to %s?" % settings.output_path, settings.assume_yes):
        try:
            with open(settings.output_path, "w", encoding="utf-8") as fh:
                fh.write((text or "").rstrip() + "\n")
            ui.note("Wrote %s" % settings.output_path)
        except OSError as exc:
            ui.error("Could not write file: %s" % exc)
    else:
        ui.note("Skipped writing file.")


def _setup_stats(settings, registry):
    """Attach the persistent leaderboard to the registry (best-effort)."""
    if not settings.enable_stats:
        return None
    try:
        from .stats import Leaderboard, default_path
        lb = Leaderboard(settings.stats_path or default_path())
        registry.leaderboard = lb
        return lb
    except Exception:
        return None


def _run_pipeline(args, settings, client, registry, ui) -> int:
    goal = " ".join(args.goal).strip() or _stdin_goal()
    if not goal:
        ui.error("No goal provided for pipeline mode.")
        return 2
    orch = Orchestrator(settings, client, registry, ui)
    if args.dry_run:
        ui.header(goal, registry.summary())
        ui.show_plan(orch._plan(goal))
        ui.note("Dry run: execution skipped.")
        return 0
    try:
        result = orch.run(goal)
    except AuthError as exc:
        ui.error("Authentication failed mid-run: %s" % exc)
        return 2
    except BudgetExceeded as exc:
        ui.error(str(exc))
        return 1
    except KeyboardInterrupt:
        ui.error("Interrupted by user.")
        return 130
    ui.deliverable(result["deliverable"])
    ui.summary(result["score"], result["requests"], result["elapsed"],
               result["models_used"], result["status"])
    if result["missing"] and result["score"] < settings.acceptance_threshold:
        ui.warn("Not fully verified. Outstanding: " + result["missing"][:300])
    _maybe_write(ui, settings, result["deliverable"])
    return 0 if result["score"] >= settings.acceptance_threshold else 3


def _run_godmode(args, settings, client, registry, ui) -> int:
    prompt = " ".join(args.goal).strip() or _stdin_goal()
    if not prompt:
        ui.error("No prompt provided for GodMode mode.")
        return 2
    router = ModelRouter(settings, client, registry)
    godmode = GodMode(settings, router, registry)
    ui.header(prompt, registry.summary())
    try:
        result = godmode.run(prompt, role=settings.agent_role,
                             width=settings.godmode_width,
                             synthesize=settings.godmode_synthesize)
    except AuthError as exc:
        ui.error("Authentication failed: %s" % exc)
        return 2
    except BudgetExceeded as exc:
        ui.error(str(exc))
        return 1
    except OpenRouterError as exc:
        ui.error("GodMode synthesis failed: %s" % exc)
        return 1
    except KeyboardInterrupt:
        ui.error("Interrupted.")
        return 130
    ui.godmode_result(result)
    _maybe_write(ui, settings, format_report(result))
    return 0 if result.successful else 3


def _build_model_consultant(settings, router, registry, ui):
    def consult(prompt: str, role: str = "all", max_models=None) -> str:
        role = role if role in (ROLE_REASONING, ROLE_CODER, ROLE_GENERAL, "all") else "all"
        total_models = len(registry.models)
        if total_models <= 0:
            return "No models are loaded for consultation."

        remaining = settings.max_total_requests - router.client.request_count
        if remaining <= 0:
            return "No request budget remains for model consultation."

        if role == "all":
            available = [m for m in registry.models if registry.available(m.id)]
            if not available:
                available = registry.models
        else:
            available = registry.select(role, n=max(1, total_models))
        requested = len(available)
        if max_models is not None:
            requested = min(requested, max(1, int(max_models)))
        else:
            # Default to a diverse, capped panel instead of every loaded model -
            # consulting hundreds of near-duplicate models burns the request
            # budget for almost no extra signal. The agent can pass max_models
            # (or the user --consult-cap) to widen it deliberately.
            requested = min(requested, max(1, settings.consult_default_models))

        # Reserve one request for synthesis when there is enough budget for it.
        budget_width = remaining - 1 if remaining > 1 else 1
        width = max(1, min(requested, budget_width))
        capped = width < requested

        ui.note("Consulting %d model(s) for collaboration%s." % (
            width, " (capped by request budget)" if capped else ""))
        result = GodMode(settings, router, registry).run(
            prompt, role=role, width=width, synthesize=True)

        lines = [
            "Model collaboration result:",
            "consulted %d model(s), %d succeeded, %.1fs elapsed" % (
                len(result.answers), len(result.successful), result.elapsed),
        ]
        if capped:
            lines.append("Note: not every loaded model was consulted because "
                         "--max-requests left only %d request(s)." % remaining)
        if result.judge_model:
            lines.append("synthesis model: " + result.judge_model)
        if result.synthesis:
            lines.extend(["", "SYNTHESIS:", result.synthesis.strip()])
        elif result.synthesis_error:
            lines.extend(["", "SYNTHESIS FAILED:", result.synthesis_error])

        model_ids = [a.model_id for a in result.successful]
        if model_ids:
            lines.extend(["", "Successful models:", ", ".join(model_ids[:24])])
            if len(model_ids) > 24:
                lines.append("... and %d more" % (len(model_ids) - 24))
        failed = [a for a in result.answers if not a.ok]
        if failed:
            lines.extend(["", "Failed/limited models:"])
            lines.extend("%s: %s" % (a.model_id, a.error) for a in failed[:12])
            if len(failed) > 12:
                lines.append("... and %d more" % (len(failed) - 12))
        return "\n".join(lines)
    return consult


def _run_agent(args, settings, client, registry, ui) -> int:
    router = ModelRouter(settings, client, registry)
    consultant = _build_model_consultant(settings, router, registry, ui)
    toolbox = ToolBox(settings, ui, root=os.path.abspath(args.root or os.getcwd()),
                      allow_outside=settings.allow_outside,
                      allow_bash=settings.allow_bash,
                      auto_approve=settings.auto_approve,
                      model_consultant=consultant)
    goal = " ".join(args.goal).strip() or _stdin_goal()
    if goal:
        agent = Agent(settings, router, ui, toolbox, root=toolbox.root)
        ui.header(goal, registry.summary())
        try:
            summary = agent.handle(goal)
        except AuthError as exc:
            ui.error("Authentication failed: %s" % exc)
            return 2
        except BudgetExceeded as exc:
            ui.error(str(exc))
            return 1
        except KeyboardInterrupt:
            ui.error("Interrupted.")
            return 130
        _maybe_write(ui, settings, summary)
        return 0
    if not sys.stdin.isatty():
        ui.error("No goal provided and no interactive terminal available.")
        return 2
    return run_repl(settings, router, registry, ui, toolbox)


def main(argv=None) -> int:
    load_dotenv()  # populate env from ./.env if present (real env wins)
    args = _build_parser().parse_args(argv)
    ui = UI(no_color=args.no_color, verbose=args.verbose)

    if args.self_test:
        from .selftest import run_self_tests
        return run_self_tests(ui)

    if args.godmode_providers:
        ui.plain(format_godmode_catalog())
        return 0

    if not _require_key(ui, args.providers or ""):
        return 2

    settings = _settings_from_args(args)
    client = OpenRouterClient(settings)
    registry = ModelRegistry(settings)
    try:
        providers = ", ".join(client.enabled_provider_names()) or "<none>"
        ui.note("Discovering free/free-tier models on: %s" % providers)
        registry.load(client)
    except AuthError as exc:
        ui.error("Authentication failed: %s" % exc)
        return 2
    except (OpenRouterError, RuntimeError) as exc:
        ui.error("Could not load models: %s" % exc)
        return 1

    for provider, warning in sorted(client.provider_errors.items()):
        ui.warn("Provider %s catalogue warning: %s" % (provider, warning[:220]))

    leaderboard = _setup_stats(settings, registry)

    if args.list_models:
        ui.models_table(registry.models, registry,
                        [ROLE_REASONING, ROLE_CODER, ROLE_GENERAL])
        return 0
    if args.stats:
        ui.leaderboard_table(leaderboard.top(20) if leaderboard else [])
        return 0

    try:
        if args.web:
            from .web import serve_web
            root = os.path.abspath(args.root or os.getcwd())
            return serve_web(settings, client, registry, root,
                             port=args.web_port or 8765)
        if args.godmode:
            return _run_godmode(args, settings, client, registry, ui)
        if args.pipeline or args.dry_run:
            return _run_pipeline(args, settings, client, registry, ui)
        return _run_agent(args, settings, client, registry, ui)
    finally:
        if leaderboard is not None:
            leaderboard.save()
