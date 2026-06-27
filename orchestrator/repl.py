"""Interactive REPL - the Claude-Code-style chat loop.

Type a goal and the agent works on it with tools; slash-commands control the
session.
"""
from __future__ import annotations

import os

from .agent import Agent
from .client import AuthError, BudgetExceeded
from .godmode import GodMode
from .registry import ROLE_CODER, ROLE_GENERAL, ROLE_REASONING

_HELP = """Commands:
  /help            show this help
  /tools           list the tools the agent can use
  /models          show selected models and routing
  /stats           show the learned model leaderboard
  /godmode PROMPT  ask several diverse models in parallel and synthesize
  /auto            toggle auto-approve for file writes & shell commands
  /undo            revert the agent's last file write/edit
  /reset           clear the conversation history
  /cwd             show the working directory
  /exit, /quit     leave

Anything else is treated as a goal for the agent to accomplish."""


def run_repl(settings, router, registry, ui, toolbox) -> int:
    agent = Agent(settings, router, ui, toolbox, root=toolbox.root)
    ui.agent_welcome(toolbox.root, registry.summary(), toolbox.auto_approve)

    while True:
        try:
            line = ui.read_prompt(os.path.basename(toolbox.root) or "ofo")
        except (EOFError, KeyboardInterrupt):
            ui.note("bye.")
            return 0

        line = line.strip()
        if not line:
            continue

        if line in ("/exit", "/quit", "/q"):
            ui.note("bye.")
            return 0
        if line in ("/help", "/?"):
            ui.plain(_HELP)
            continue
        if line == "/tools":
            ui.plain(toolbox.spec())
            continue
        if line == "/models":
            ui.models_table(registry.models, registry,
                            [ROLE_REASONING, ROLE_CODER, ROLE_GENERAL])
            continue
        if line == "/auto":
            state = toolbox.toggle_auto()
            ui.note("auto-approve is now %s." % ("ON" if state else "OFF"))
            continue
        if line == "/undo":
            res = toolbox.undo()
            (ui.note if res.ok else ui.warn)(res.output)
            continue
        if line == "/stats":
            lb = getattr(registry, "leaderboard", None)
            ui.leaderboard_table(lb.top(12) if lb else [])
            continue
        if (line == "/godmode" or line.startswith("/godmode ")
                or line == "/gm" or line.startswith("/gm ")):
            parts = line.split(" ", 1)
            prompt = parts[1].strip() if len(parts) > 1 else ""
            if not prompt:
                ui.warn("usage: /godmode <prompt>")
                continue
            try:
                result = GodMode(settings, router, registry).run(
                    prompt, role=settings.agent_role,
                    width=settings.godmode_width,
                    synthesize=settings.godmode_synthesize)
                ui.godmode_result(result)
            except AuthError as exc:
                ui.error("Authentication failed: %s" % exc)
                return 2
            except BudgetExceeded as exc:
                ui.error(str(exc))
                ui.note("Raise the cap with --max-requests and restart.")
                return 1
            except KeyboardInterrupt:
                ui.warn("interrupted - back to prompt.")
            continue
        if line == "/reset":
            agent.reset()
            continue
        if line == "/cwd":
            ui.note(toolbox.root)
            continue
        if line.startswith("/"):
            ui.warn("unknown command: %s (try /help)" % line)
            continue

        try:
            agent.handle(line)
        except AuthError as exc:
            ui.error("Authentication failed: %s" % exc)
            return 2
        except BudgetExceeded as exc:
            ui.error(str(exc))
            ui.note("Raise the cap with --max-requests and restart.")
            return 1
        except KeyboardInterrupt:
            ui.warn("interrupted - back to prompt.")
            continue
