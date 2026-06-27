"""A zero-dependency graphical web UI for the orchestrator.

Runs a local http.server, opens the browser, and streams the agent's live
activity (thoughts, tool calls, observations, model collaboration, approvals)
to a self-contained single-page app over Server-Sent Events. The agent runs in
a background thread; a `WebUI` adapter implements the same interface the CLI
`UI` exposes, but turns every event into a structured message for the browser.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .agent import Agent
from .router import ModelRouter
from .tools import ToolBox


# --------------------------------------------------------------------------- #
# Event bus: one publisher (the agent thread), many SSE subscribers.
# --------------------------------------------------------------------------- #
class EventBus:
    def __init__(self, history: int = 300):
        self._subs = []
        self._history = []
        self._history_max = history
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            for ev in self._history:      # replay so a fresh tab sees the run so far
                q.put(ev)
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
            subs = list(self._subs)
        for q in subs:
            q.put(event)

    def clear_history(self) -> None:
        with self._lock:
            self._history = []


# --------------------------------------------------------------------------- #
# WebUI: same surface as the terminal UI, but emits events instead of printing.
# --------------------------------------------------------------------------- #
class WebUI:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self._pending = {}     # approval id -> (threading.Event, holder dict)
        self._lock = threading.Lock()
        self._next_id = 0

    def _emit(self, **event) -> None:
        self.bus.publish(event)

    # -- methods the Agent calls ------------------------------------------ #
    def agent_step(self, step, model, thought, tool, args) -> None:
        self._emit(type="step", step=step, model=model, thought=thought,
                   tool=tool, args=args if isinstance(args, dict) else {})

    def agent_observation(self, tool, ok, output) -> None:
        self._emit(type="observation", tool=tool, ok=bool(ok), output=str(output))

    def agent_finish(self, summary) -> None:
        self._emit(type="finish", summary=str(summary))

    def agent_usage(self, requests, cache_hits, models, elapsed) -> None:
        self._emit(type="usage", requests=requests, cache_hits=cache_hits,
                   models=list(models), elapsed=round(float(elapsed), 1))

    # -- methods the ToolBox calls ---------------------------------------- #
    def tool_preview(self, title, body) -> None:
        self._emit(type="preview", title=str(title), body=str(body)[:2000])

    def note(self, text) -> None:
        self._emit(type="note", level="note", text=str(text))

    def warn(self, text) -> None:
        self._emit(type="note", level="warn", text=str(text))

    def error(self, text) -> None:
        self._emit(type="note", level="error", text=str(text))

    def confirm(self, summary, assume_yes=False) -> bool:
        """Ask the browser to approve an action and block until it answers."""
        if assume_yes:
            return True
        with self._lock:
            self._next_id += 1
            aid = self._next_id
            ev = threading.Event()
            holder = {"approved": False}
            self._pending[aid] = (ev, holder)
        self._emit(type="approval", id=aid, summary=str(summary))
        answered = ev.wait(timeout=300)   # deny on timeout (safe default)
        with self._lock:
            self._pending.pop(aid, None)
        return bool(holder["approved"]) if answered else False

    def resolve_approval(self, aid, approved) -> bool:
        with self._lock:
            item = self._pending.get(aid)
        if not item:
            return False
        ev, holder = item
        holder["approved"] = bool(approved)
        ev.set()
        self._emit(type="approval_done", id=aid, approved=bool(approved))
        return True

    # Any other UI method the (evolving) codebase might call becomes a no-op,
    # so the web run never crashes on an unimplemented bit of UI surface.
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def _noop(*_a, **_k):
            return None
        return _noop


# --------------------------------------------------------------------------- #
# Application state shared with the HTTP handlers.
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, settings, registry, root):
        self.settings = settings
        self.registry = registry
        self.root = root
        self.bus = EventBus()
        self.ui = WebUI(self.bus)
        self.router = None
        self.toolbox = None
        self.agent = None
        self.leaderboard = getattr(registry, "leaderboard", None)
        self.run_lock = threading.Lock()
        self.busy = False

    # -- payloads --------------------------------------------------------- #
    def models_payload(self) -> dict:
        models = self.registry.models
        providers = {}
        for m in models:
            providers[m.provider] = providers.get(m.provider, 0) + 1
        leaderboard = []
        if self.leaderboard is not None:
            for mid, rec in self.leaderboard.top(12):
                leaderboard.append({
                    "id": mid, "ok": rec.get("ok", 0), "fail": rec.get("fail", 0),
                    "latency": round(rec.get("ema_latency", 0) or 0, 1),
                })
        routing = {}
        for role in ("reasoning", "coder", "general"):
            try:
                routing[role] = [m.id for m in self.registry.select(role, n=5)]
            except Exception:
                routing[role] = []
        return {
            "count": len(models),
            "providers": providers,
            "leaderboard": leaderboard,
            "routing": routing,
            "cwd": self.root,
            "auto": bool(getattr(self.toolbox, "auto_approve", False)),
            "busy": self.busy,
        }

    # -- run -------------------------------------------------------------- #
    def start_run(self, goal: str) -> bool:
        goal = (goal or "").strip()
        if not goal:
            return False
        with self.run_lock:
            if self.busy:
                return False
            self.busy = True

        def worker():
            self.bus.publish({"type": "run_start", "goal": goal})
            try:
                self.agent.handle(goal)
            except Exception as exc:  # noqa: BLE001 - surface, never crash server
                self.bus.publish({"type": "note", "level": "error",
                                  "text": "run failed: %s" % exc})
            finally:
                self.busy = False
                self.bus.publish({"type": "done"})

        threading.Thread(target=worker, daemon=True).start()
        return True


def build_app(settings, client, registry, root: str) -> App:
    """Wire the agent stack onto a WebUI. Imported lazily by the CLI."""
    from .cli import _build_model_consultant  # lazy: avoids an import cycle

    app = App(settings, registry, root)
    app.router = ModelRouter(settings, client, registry)
    consultant = _build_model_consultant(settings, app.router, registry, app.ui)
    app.toolbox = ToolBox(settings, app.ui, root=root,
                          allow_outside=settings.allow_outside,
                          allow_bash=settings.allow_bash,
                          auto_approve=settings.auto_approve,
                          model_consultant=consultant)
    app.agent = Agent(settings, app.router, app.ui, app.toolbox, root=root)
    return app


# --------------------------------------------------------------------------- #
# HTTP + SSE handler
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "ofo-web"

    def log_message(self, *_a):   # keep the terminal clean
        return

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"))

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            return json.loads(raw or b"{}")
        except (ValueError, TypeError):
            return {}

    # -- GET -------------------------------------------------------------- #
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/models":
            self._json(self.app.models_payload())
        elif path == "/api/events":
            self._sse()
        else:
            self._send(404, b"not found", "text/plain")

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.app.bus.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(ev, default=str, ensure_ascii=False)
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.bus.unsubscribe(q)

    # -- POST ------------------------------------------------------------- #
    def do_POST(self):
        path = urlparse(self.path).path
        data = self._body()
        if path == "/api/run":
            ok = self.app.start_run(data.get("goal", ""))
            self._json({"ok": ok, "busy": self.app.busy})
        elif path == "/api/approve":
            ok = self.app.ui.resolve_approval(data.get("id"), data.get("approve"))
            self._json({"ok": ok})
        elif path == "/api/auto":
            if self.app.toolbox is not None:
                self.app.toolbox.auto_approve = bool(data.get("on"))
            self._json({"ok": True, "auto": bool(data.get("on"))})
        else:
            self._send(404, b"not found", "text/plain")


def serve(app: App, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.app = app  # type: ignore[attr-defined]
    httpd.daemon_threads = True
    if open_browser:
        try:
            webbrowser.open("http://%s:%d/" % (host, port))
        except Exception:
            pass
    return httpd


def serve_web(settings, client, registry, root: str,
              host: str = "127.0.0.1", port: int = 8765) -> int:
    app = build_app(settings, client, registry, root)
    httpd = serve(app, host, port, open_browser=True)
    url = "http://%s:%d/" % (host, port)
    print("\n  🌐  Free-Tier AI Orchestrator — web UI running at %s" % url)
    print("      working directory: %s" % root)
    print("      press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping web UI…")
    finally:
        httpd.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# The single-page app (self-contained: no external assets, no build step).
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Free-Tier AI Orchestrator</title>
<style>
:root{
  --bg:#0a0e17; --bg2:#0f1626; --panel:rgba(255,255,255,.045);
  --panel-bd:rgba(255,255,255,.09); --txt:#e8edf6; --mut:#8a97ad;
  --acc:#6ea8fe; --acc2:#a78bfa; --ok:#41d6a0; --warn:#f7b955; --err:#ff6b6b;
  --grad:linear-gradient(135deg,#6ea8fe,#a78bfa 60%,#f08fc0);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  font-family:'Inter',system-ui,Segoe UI,Roboto,sans-serif;color:var(--txt);
  background:
    radial-gradient(1200px 700px at 85% -10%,rgba(167,139,250,.18),transparent 60%),
    radial-gradient(1000px 600px at -10% 110%,rgba(110,168,254,.16),transparent 55%),
    var(--bg);
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}
header{
  display:flex;align-items:center;gap:14px;padding:14px 22px;
  border-bottom:1px solid var(--panel-bd);backdrop-filter:blur(8px);
}
.logo{font-size:24px;filter:drop-shadow(0 0 10px rgba(110,168,254,.5))}
.title{font-weight:700;font-size:17px;letter-spacing:.2px}
.title b{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);font-size:12px}
.spacer{flex:1}
.pill{font-size:12px;color:var(--mut);border:1px solid var(--panel-bd);
  padding:6px 12px;border-radius:999px;background:var(--panel)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mut);margin-right:6px;vertical-align:1px}
.dot.live{background:var(--ok);box-shadow:0 0 10px var(--ok);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
main{display:flex;flex:1;min-height:0}
aside{
  width:312px;flex:none;border-right:1px solid var(--panel-bd);
  padding:18px;overflow:auto;background:linear-gradient(180deg,rgba(255,255,255,.02),transparent)
}
.card{background:var(--panel);border:1px solid var(--panel-bd);border-radius:14px;padding:14px;margin-bottom:14px}
.card h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:4px 0}
.kv .v{color:var(--mut);font-variant-numeric:tabular-nums}
.prov{display:flex;align-items:center;gap:8px;font-size:13px;padding:5px 0}
.prov .pdot{width:9px;height:9px;border-radius:50%;background:var(--acc)}
.bar{height:6px;border-radius:5px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:3px}
.bar>span{display:block;height:100%;background:var(--grad)}
.lb-row{font-size:12px;margin-bottom:9px}
.lb-row .mid{color:var(--txt);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lb-row .meta{color:var(--mut);font-size:11px}
.route{font-size:12px;margin-bottom:8px}
.route .role{color:var(--acc2);font-weight:600;text-transform:capitalize}
.route .m{color:var(--mut);display:block;margin-left:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.content{flex:1;display:flex;flex-direction:column;min-width:0}
#timeline{flex:1;overflow:auto;padding:24px 26px 8px;scroll-behavior:smooth}
.welcome{max-width:680px;margin:6vh auto 0;text-align:center}
.welcome h1{font-size:30px;margin:0 0 10px}
.welcome h1 span{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.welcome p{color:var(--mut);font-size:14px;line-height:1.7}
.chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:20px}
.chip{cursor:pointer;font-size:13px;color:var(--txt);background:var(--panel);
  border:1px solid var(--panel-bd);padding:9px 14px;border-radius:11px;transition:.15s}
.chip:hover{border-color:var(--acc);transform:translateY(-1px)}
.item{animation:rise .28s ease both;margin:0 auto 14px;max-width:920px}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.step{background:var(--panel);border:1px solid var(--panel-bd);border-left:3px solid var(--acc);
  border-radius:13px;padding:13px 16px}
.step .head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.num{font-size:11px;font-weight:700;color:#0a0e17;background:var(--grad);
  width:22px;height:22px;border-radius:7px;display:flex;align-items:center;justify-content:center}
.model{font-size:11px;color:var(--mut);border:1px solid var(--panel-bd);padding:2px 8px;border-radius:7px}
.thought{color:var(--mut);font-size:13px;margin:2px 0 0;line-height:1.5}
.toolrow{display:flex;align-items:center;gap:9px;margin-top:9px;flex-wrap:wrap}
.tbadge{font-size:12.5px;font-weight:600;padding:5px 11px;border-radius:9px;display:inline-flex;gap:7px;align-items:center}
.tb-read{background:rgba(110,168,254,.16);color:#9cc2ff}
.tb-write{background:rgba(167,139,250,.18);color:#c4b1ff}
.tb-bash{background:rgba(247,185,85,.16);color:#ffd591}
.tb-think{background:rgba(65,214,160,.16);color:#84e9c5}
.tb-audio{background:rgba(240,143,192,.18);color:#ffb4dd}
.tb-finish{background:rgba(65,214,160,.2);color:#84e9c5}
.targ{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--mut);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.preview{margin:10px 0 2px}
.preview pre{margin:0;background:#070b13;border:1px solid var(--panel-bd);border-radius:10px;
  padding:11px 13px;font-size:12px;color:#b9c4d6;max-height:230px;overflow:auto;white-space:pre-wrap}
.obs{max-width:920px;margin:-6px auto 14px;padding:9px 15px 9px 18px;font-size:12.5px;
  font-family:ui-monospace,Menlo,monospace;border-radius:0 10px 10px 0;border-left:2px solid var(--mut);
  color:var(--mut);background:rgba(255,255,255,.025)}
.obs.ok{border-left-color:var(--ok)} .obs.err{border-left-color:var(--err);color:#ffb4b4}
.note{max-width:920px;margin:0 auto 12px;font-size:12.5px;color:var(--mut)}
.note.warn{color:var(--warn)} .note.error{color:var(--err)}
.finish{max-width:920px;margin:6px auto 16px;background:linear-gradient(135deg,rgba(65,214,160,.14),rgba(110,168,254,.1));
  border:1px solid rgba(65,214,160,.35);border-radius:14px;padding:16px 18px}
.finish .t{font-weight:700;color:var(--ok);margin-bottom:6px;display:flex;gap:8px;align-items:center}
.finish .b{color:var(--txt);font-size:14px;line-height:1.6}
.composer{padding:14px 26px 18px;border-top:1px solid var(--panel-bd);background:rgba(8,11,18,.6);backdrop-filter:blur(8px)}
.cbar{max-width:920px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;resize:none;background:var(--panel);border:1px solid var(--panel-bd);color:var(--txt);
  border-radius:13px;padding:13px 15px;font:inherit;font-size:14px;min-height:24px;max-height:160px;outline:none}
textarea:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(110,168,254,.14)}
.send{background:var(--grad);color:#0a0e17;font-weight:700;border:none;border-radius:13px;
  padding:0 20px;height:50px;cursor:pointer;font-size:14px;transition:.15s}
.send:disabled{opacity:.45;cursor:not-allowed}
.send:not(:disabled):hover{filter:brightness(1.08);transform:translateY(-1px)}
.toggle{display:flex;align-items:center;gap:9px;font-size:12px;color:var(--mut);margin-top:10px;max-width:920px;margin-left:auto;margin-right:auto}
.sw{width:38px;height:21px;border-radius:999px;background:rgba(255,255,255,.14);position:relative;cursor:pointer;transition:.2s}
.sw.on{background:var(--ok)}
.sw::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;background:#fff;transition:.2s}
.sw.on::after{left:19px}
.typing{display:inline-flex;gap:4px;align-items:center}
.typing i{width:6px;height:6px;border-radius:50%;background:var(--acc);animation:bnc 1s infinite}
.typing i:nth-child(2){animation-delay:.15s} .typing i:nth-child(3){animation-delay:.3s}
@keyframes bnc{0%,100%{opacity:.3;transform:translateY(0)}50%{opacity:1;transform:translateY(-4px)}}
.overlay{position:fixed;inset:0;background:rgba(3,6,12,.66);backdrop-filter:blur(3px);
  display:none;align-items:center;justify-content:center;z-index:50}
.overlay.show{display:flex}
.modal{background:#0f1626;border:1px solid var(--panel-bd);border-radius:16px;padding:22px;max-width:560px;width:90%}
.modal h2{margin:0 0 6px;font-size:16px}
.modal .det{color:var(--mut);font-size:13px;margin:10px 0 18px;font-family:ui-monospace,Menlo,monospace;
  background:#070b13;border:1px solid var(--panel-bd);border-radius:10px;padding:12px;max-height:200px;overflow:auto;white-space:pre-wrap}
.modal .row{display:flex;gap:10px;justify-content:flex-end}
.btn{border:none;border-radius:10px;padding:10px 18px;font-weight:600;cursor:pointer;font-size:13px}
.btn.ok{background:var(--grad);color:#0a0e17} .btn.no{background:rgba(255,255,255,.08);color:var(--txt)}
::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:9px}
</style>
</head>
<body>
<header>
  <span class="logo">🤖</span>
  <div>
    <div class="title">Free-Tier AI <b>Orchestrator</b></div>
    <div class="sub" id="cwd">…</div>
  </div>
  <div class="spacer"></div>
  <div class="pill"><span class="dot" id="statedot"></span><span id="statetxt">idle</span></div>
  <div class="pill" id="modelpill">…</div>
</header>
<main>
  <aside>
    <div class="card">
      <h3>Live usage</h3>
      <div class="kv"><span>Requests</span><span class="v" id="u-req">0</span></div>
      <div class="kv"><span>Cache hits</span><span class="v" id="u-cache">0</span></div>
      <div class="kv"><span>Elapsed</span><span class="v" id="u-time">0.0s</span></div>
      <div class="kv"><span>Models used</span><span class="v" id="u-models">—</span></div>
    </div>
    <div class="card"><h3>Providers</h3><div id="providers"></div></div>
    <div class="card"><h3>Routing</h3><div id="routing"></div></div>
    <div class="card"><h3>Model leaderboard</h3><div id="leaderboard"></div></div>
  </aside>
  <section class="content">
    <div id="timeline">
      <div class="welcome" id="welcome">
        <h1>Tell the <span>AI team</span> what to do</h1>
        <p>It plans, reads &amp; writes files, runs commands and lets diverse free models
           collaborate — all in your current folder, with your approval. Watch every step live.</p>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <div class="composer">
      <div class="cbar">
        <textarea id="goal" placeholder="e.g. add a --json flag to cli.py and update the tests…" rows="1"></textarea>
        <button class="send" id="send">Run ▸</button>
      </div>
      <div class="toggle">
        <div class="sw" id="autosw"></div><span id="autotxt">Auto-approve actions</span>
        <span class="spacer" style="flex:1"></span>
        <span id="hint">Shift+Enter for newline</span>
      </div>
    </div>
  </section>
</main>
<div class="overlay" id="overlay">
  <div class="modal">
    <h2>⚠️ Approve this action?</h2>
    <div class="det" id="approve-det"></div>
    <div class="row">
      <button class="btn no" id="deny">Deny</button>
      <button class="btn ok" id="approve">Approve</button>
    </div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s), el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const TL=$('#timeline'), GOAL=$('#goal'), SEND=$('#send');
const ICON={read_file:'📄',list_dir:'📁',write_file:'✏️',edit_file:'✏️',run_bash:'⚡',consult_models:'🧠',master_audio:'🎚️',finish:'✅'};
const CLS={read_file:'tb-read',list_dir:'tb-read',write_file:'tb-write',edit_file:'tb-write',run_bash:'tb-bash',consult_models:'tb-think',master_audio:'tb-audio',finish:'tb-finish'};
let running=false, started=0, autoApprove=false;
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function argText(tool,a){if(!a)return'';return a.command||a.prompt||a.path||a.summary||(()=>{try{return JSON.stringify(a).slice(0,180)}catch(e){return''}})();}
function autoscroll(){TL.scrollTop=TL.scrollHeight;}
function clearWelcome(){const w=$('#welcome');if(w)w.remove();}

function addStep(ev){
  clearWelcome();
  const wrap=el('div','item');
  const s=el('div','step');
  const head=el('div','head');
  head.appendChild(el('span','num',esc(ev.step)));
  if(ev.model)head.appendChild(el('span','model',esc(ev.model)));
  s.appendChild(head);
  if(ev.thought)s.appendChild(el('div','thought',esc(ev.thought)));
  const tr=el('div','toolrow');
  tr.appendChild(el('span','tbadge '+(CLS[ev.tool]||'tb-read'),(ICON[ev.tool]||'🔧')+' '+esc(ev.tool)));
  const at=argText(ev.tool,ev.args); if(at)tr.appendChild(el('span','targ',esc(at)));
  s.appendChild(tr); wrap.appendChild(s); TL.appendChild(wrap); autoscroll();
}
function addPreview(ev){clearWelcome();const w=el('div','item');const p=el('div','preview');
  const pre=el('pre',null,esc((ev.title?ev.title+'\n':'')+ev.body));p.appendChild(pre);w.appendChild(p);TL.appendChild(w);autoscroll();}
function addObs(ev){const o=el('div','obs '+(ev.ok?'ok':'err'),(ev.ok?'✓ ':'✗ ')+esc((ev.output||'').split('\n')[0].slice(0,260)));TL.appendChild(o);autoscroll();}
function addNote(ev){const lv=ev.level==='warn'?'warn':ev.level==='error'?'error':'';TL.appendChild(el('div','note '+lv,esc(ev.text)));autoscroll();}
function addFinish(ev){clearWelcome();const f=el('div','finish');f.appendChild(el('div','t','✅ Done'));f.appendChild(el('div','b',esc(ev.summary)));TL.appendChild(f);autoscroll();}
function setRunning(on){running=on;SEND.disabled=on;$('#statedot').className='dot'+(on?' live':'');$('#statetxt').textContent=on?'working…':'idle';
  if(on){started=Date.now();}else{loadModels();}}
function updUsage(ev){$('#u-req').textContent=ev.requests;$('#u-cache').textContent=ev.cache_hits;
  $('#u-time').textContent=ev.elapsed+'s';const u=[...new Set((ev.models||[]).map(m=>m.replace('cache:','')))];
  $('#u-models').textContent=u.length?u.length:'—';}

function showApproval(ev){$('#approve-det').textContent=ev.summary;$('#overlay').classList.add('show');
  const done=ok=>{$('#overlay').classList.remove('show');fetch('/api/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:ev.id,approve:ok})});};
  $('#approve').onclick=()=>done(true);$('#deny').onclick=()=>done(false);}

const EH={run_start:e=>{setRunning(true);TL.querySelectorAll('.item,.obs,.note,.finish').forEach(n=>0);},
  step:addStep,preview:addPreview,observation:addObs,note:addNote,finish:addFinish,
  usage:updUsage,approval:showApproval,done:()=>setRunning(false),approval_done:()=>$('#overlay').classList.remove('show')};
function handle(ev){(EH[ev.type]||(()=>{}))(ev);}

function send(){const g=GOAL.value.trim();if(!g||running)return;GOAL.value='';GOAL.style.height='auto';
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal:g})})
   .then(r=>r.json()).then(d=>{if(!d.ok&&!d.busy)addNote({level:'warn',text:'could not start run'});});}
SEND.onclick=send;
GOAL.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
GOAL.addEventListener('input',()=>{GOAL.style.height='auto';GOAL.style.height=Math.min(GOAL.scrollHeight,160)+'px';});

function setAuto(on){autoApprove=on;$('#autosw').className='sw'+(on?' on':'');
  $('#autotxt').textContent=on?'Auto-approve: ON':'Auto-approve actions';}
$('#autosw').onclick=()=>{const on=!autoApprove;setAuto(on);
  fetch('/api/auto',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on})});};

function loadModels(){fetch('/api/models').then(r=>r.json()).then(d=>{
  $('#cwd').textContent=d.cwd; $('#modelpill').textContent=d.count+' models · '+Object.keys(d.providers).length+' providers';
  setAuto(!!d.auto);
  const P=$('#providers');P.innerHTML='';Object.entries(d.providers).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{
    const r=el('div','prov');r.appendChild(el('span','pdot'));r.appendChild(el('span',null,esc(k)));
    const sp=el('span','spacer');sp.style.flex='1';r.appendChild(sp);r.appendChild(el('span','meta',v+''));P.appendChild(r);});
  const R=$('#routing');R.innerHTML='';Object.entries(d.routing).forEach(([role,ms])=>{
    const r=el('div','route');r.appendChild(el('span','role',role));
    r.appendChild(el('span','m',esc((ms[0]||'—'))));R.appendChild(r);});
  const L=$('#leaderboard');L.innerHTML='';if(!d.leaderboard.length){L.innerHTML='<div class="meta" style="color:var(--mut);font-size:12px">No stats yet — run a task.</div>';}
  d.leaderboard.forEach(m=>{const tot=m.ok+m.fail||1;const row=el('div','lb-row');
    row.appendChild(el('div','mid',esc(m.id)));
    row.appendChild(el('div','meta','ok '+m.ok+' · fail '+m.fail+(m.latency?' · '+m.latency+'s':'')));
    const b=el('div','bar');b.appendChild(el('span',null,''));b.firstChild.style.width=Math.round(m.ok/tot*100)+'%';
    row.appendChild(b);L.appendChild(row);});
});}

const CHIPS=['Summarise every file in this folder','Write a Python script that renames files by date',
  'Create a README for this project','Explain what the code here does, simply'];
const C=$('#chips');CHIPS.forEach(t=>{const c=el('div','chip',t);c.onclick=()=>{GOAL.value=t;GOAL.focus();GOAL.dispatchEvent(new Event('input'));};C.appendChild(c);});

const ev=new EventSource('/api/events');
ev.onmessage=e=>{try{handle(JSON.parse(e.data));}catch(x){}};
loadModels();setInterval(()=>{if(running){$('#u-time').textContent=((Date.now()-started)/1000).toFixed(1)+'s';}},200);
</script>
</body>
</html>"""
