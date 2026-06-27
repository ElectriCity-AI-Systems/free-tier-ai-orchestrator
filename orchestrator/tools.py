"""Tools the agent can call to actually *do* things: inspect the project,
read/write/edit files, master audio, and run shell commands.

Safety is built in, not bolted on:
  * file access is sandboxed to the working directory by default,
  * side-effecting tools (write/edit/master_audio/run_bash) require confirmation,
  * obviously destructive shell commands always require confirmation
    (even in --auto mode),
  * all observations are size-capped so a huge file/output can't blow up
    the model's context or the terminal.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Optional

# Patterns that must never be auto-approved, even with --auto.
_DANGEROUS = re.compile(
    r"(\brm\s+-[a-z]*\s*[rf]|\bmkfs\b|\bdd\s+if=|>\s*/dev/sd|:\(\)\s*\{|"
    r"\bshutdown\b|\breboot\b|\bchmod\s+-R\s+777\s+/|\bgit\s+push\b|"
    r"\bcurl\b.*\|\s*(ba)?sh|\bwget\b.*\|\s*(ba)?sh)",
    re.IGNORECASE,
)


@dataclass
class ToolResult:
    ok: bool
    output: str


class ToolError(Exception):
    pass


TOOL_SPEC = """- list_dir   {"path": "."}                         list files in a directory
- read_file  {"path": "file.py"}                   read a UTF-8 text file
- write_file {"path": "f.py", "content": "..."}    create/overwrite a file
- edit_file  {"path": "f.py", "find": "...", "replace": "..."}  replace first exact match
- run_bash   {"command": "pytest -q"}              run a shell command in the working dir
- consult_models {"prompt": "ask several diverse models for strategy", "role": "all", "max_models": 8}  consult a diverse panel of models (default ~8) and synthesize advice; raise max_models to widen
- master_audio {"path": "track.wav", "output": "track_tunecore_master.wav", "profile": "tunecore"}  create a mastered 44.1k/16-bit stereo WAV with ffmpeg
- finish     {"summary": "what was accomplished"}  stop; you are done"""


_AUDIO_PROFILES = {
    # TuneCore's public distribution guide asks for 16-bit, 44.1 kHz, stereo
    # WAV. The loudness defaults are conservative streaming-safe values.
    "tunecore": {"target_lufs": -14.0, "true_peak": -1.0, "lra": 11.0,
                 "sample_rate": 44100, "bit_depth": 16, "channels": 2},
    "streaming": {"target_lufs": -14.0, "true_peak": -1.0, "lra": 11.0,
                  "sample_rate": 44100, "bit_depth": 16, "channels": 2},
    "dynamic": {"target_lufs": -16.0, "true_peak": -1.5, "lra": 12.0,
                "sample_rate": 44100, "bit_depth": 16, "channels": 2},
    "loud": {"target_lufs": -11.0, "true_peak": -1.0, "lra": 8.0,
             "sample_rate": 44100, "bit_depth": 16, "channels": 2},
}

_PCM_CODECS = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}


class ToolBox:
    def __init__(self, settings, ui, root: str,
                 allow_outside: bool = False, allow_bash: bool = True,
                 auto_approve: bool = False,
                 model_consultant: Optional[Callable] = None):
        self.s = settings
        self.ui = ui
        self.root = os.path.abspath(root)
        self.allow_outside = allow_outside
        self.allow_bash = allow_bash
        self.auto_approve = auto_approve
        self.model_consultant = model_consultant
        self._undo_stack = []  # (abs_path, previous_content_or_None) for /undo

    # -- metadata ---------------------------------------------------------- #
    def names(self):
        base = {"list_dir", "read_file", "write_file", "edit_file",
                "master_audio", "finish"}
        if self.allow_bash:
            base.add("run_bash")
        if self.model_consultant is not None:
            base.add("consult_models")
        return base

    def spec(self) -> str:
        lines = []
        for line in TOOL_SPEC.splitlines():
            if "run_bash" in line and not self.allow_bash:
                continue
            if "consult_models" in line and self.model_consultant is None:
                continue
            lines.append(line)
        return "\n".join(lines)

    def toggle_auto(self) -> bool:
        self.auto_approve = not self.auto_approve
        return self.auto_approve

    # -- helpers ----------------------------------------------------------- #
    def _resolve(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ToolError("missing 'path'")
        full = os.path.abspath(os.path.join(self.root, os.path.expanduser(path)))
        if not self.allow_outside:
            if full != self.root and not full.startswith(self.root + os.sep):
                raise ToolError("path '%s' escapes the working directory "
                                "(use --allow-outside to permit)" % path)
        return full

    def _truncate(self, text: str) -> str:
        cap = self.s.max_tool_output
        if len(text) <= cap:
            return text
        return text[:cap] + "\n… [truncated %d more chars]" % (len(text) - cap)

    def _approve(self, summary: str) -> bool:
        if self.auto_approve:
            self.ui.note("[auto-approved] " + summary)
            return True
        return self.ui.confirm(summary, assume_yes=False)

    def _rel(self, full: str) -> str:
        try:
            return os.path.relpath(full, self.root)
        except ValueError:
            return full

    def _snapshot(self, full: str):
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None

    # -- undo / introspection (used by the REPL and the agent read-cache) -- #
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def undo(self) -> ToolResult:
        if not self._undo_stack:
            return ToolResult(False, "nothing to undo")
        full, prev = self._undo_stack.pop()
        rel = self._rel(full)
        try:
            if prev is None:
                if os.path.isfile(full):
                    os.remove(full)
                return ToolResult(True, "undid creation of %s (removed)" % rel)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(prev)
            return ToolResult(True, "reverted %s to its previous contents" % rel)
        except OSError as exc:
            return ToolResult(False, "undo failed: %s" % exc)

    def stat(self, path: str):
        """Sandbox-resolved stat for cache decisions; (abs, mtime, size) or None."""
        try:
            full = self._resolve(path)
        except ToolError:
            return None
        try:
            st = os.stat(full)
        except OSError:
            return None
        return full, st.st_mtime, st.st_size

    # -- dispatch ---------------------------------------------------------- #
    def run(self, name: str, args: Dict) -> ToolResult:
        args = args or {}
        try:
            handler = {
                "list_dir": self._list_dir,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "edit_file": self._edit_file,
                "consult_models": self._consult_models,
                "master_audio": self._master_audio,
                "run_bash": self._run_bash,
            }.get(name)
            if handler is None:
                return ToolResult(False, "unknown tool '%s'" % name)
            return handler(args)
        except ToolError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:  # never let a tool crash the agent loop
            return ToolResult(False, "%s: %s" % (type(exc).__name__, exc))

    # -- tools ------------------------------------------------------------- #
    def _list_dir(self, args) -> ToolResult:
        full = self._resolve(args.get("path", "."))
        if not os.path.isdir(full):
            return ToolResult(False, "not a directory: %s" % args.get("path", "."))
        entries = []
        for name in sorted(os.listdir(full)):
            p = os.path.join(full, name)
            entries.append((name + "/") if os.path.isdir(p) else name)
        return ToolResult(True, self._truncate("\n".join(entries) or "(empty)"))

    def _read_file(self, args) -> ToolResult:
        full = self._resolve(args.get("path", ""))
        if not os.path.isfile(full):
            return ToolResult(False, "no such file: %s" % args.get("path"))
        if os.path.getsize(full) > 2_000_000:
            return ToolResult(False, "file too large to read (>2MB)")
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return ToolResult(True, self._truncate(fh.read()))

    def _write_file(self, args) -> ToolResult:
        full = self._resolve(args.get("path", ""))
        content = args.get("content")
        if content is None:
            return ToolResult(False, "missing 'content'")
        content = str(content)
        rel = self._rel(full)
        exists = os.path.isfile(full)
        verb = "overwrite" if exists else "create"
        self.ui.tool_preview("%s %s (%d bytes)" % (verb, rel, len(content)),
                             content[:600])
        if not self._approve("%s file %s (%d bytes)?" % (verb, rel, len(content))):
            return ToolResult(False, "permission denied by user")
        prev = self._snapshot(full) if exists else None
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        self._undo_stack.append((full, prev))
        return ToolResult(True, "%sd %s (%d bytes) [undoable]" % (verb, rel, len(content)))

    def _edit_file(self, args) -> ToolResult:
        full = self._resolve(args.get("path", ""))
        find = args.get("find")
        replace = args.get("replace")
        if find is None or replace is None:
            return ToolResult(False, "edit_file needs 'find' and 'replace'")
        if not os.path.isfile(full):
            return ToolResult(False, "no such file: %s" % args.get("path"))
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            original = fh.read()
        if find not in original:
            return ToolResult(False, "'find' text not found; read the file first")
        rel = self._rel(full)
        updated = original.replace(find, replace, 1)
        self.ui.tool_preview("edit %s" % rel,
                             "- %s\n+ %s" % (str(find)[:300], str(replace)[:300]))
        if not self._approve("apply edit to %s?" % rel):
            return ToolResult(False, "permission denied by user")
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(updated)
        self._undo_stack.append((full, original))
        return ToolResult(True, "edited %s (1 replacement) [undoable]" % rel)

    def _run_bash(self, args) -> ToolResult:
        if not self.allow_bash:
            return ToolResult(False, "shell access is disabled (--no-bash)")
        command = args.get("command")
        if not command or not isinstance(command, str):
            return ToolResult(False, "missing 'command'")
        dangerous = bool(_DANGEROUS.search(command))
        self.ui.tool_preview("run_bash" + (" ⚠ DANGEROUS" if dangerous else ""), command)
        # Dangerous commands always require an explicit prompt, ignoring --auto.
        if dangerous:
            if not self.ui.confirm("Run POTENTIALLY DESTRUCTIVE command? -> %s"
                                   % command, assume_yes=False):
                return ToolResult(False, "permission denied (dangerous command)")
        elif not self._approve("run: %s" % command):
            return ToolResult(False, "permission denied by user")
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=self.s.bash_timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "command timed out after %.0fs" % self.s.bash_timeout)
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        out = self._truncate(out.strip() or "(no output)")
        return ToolResult(proc.returncode == 0,
                          "exit=%d\n%s" % (proc.returncode, out))

    # -- model collaboration ---------------------------------------------- #
    def _consult_models(self, args) -> ToolResult:
        if self.model_consultant is None:
            return ToolResult(False, "model consultation is not available")
        prompt = (args.get("prompt") or args.get("question") or args.get("task") or "")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult(False, "consult_models needs a non-empty 'prompt'")
        role = str(args.get("role") or "all").strip().lower()
        max_models = args.get("max_models")
        try:
            max_models = int(max_models) if max_models not in (None, "") else None
        except (TypeError, ValueError):
            return ToolResult(False, "max_models must be an integer when provided")
        if max_models is not None and max_models < 1:
            return ToolResult(False, "max_models must be >= 1")

        summary = "consult all available models"
        if max_models is not None:
            summary += " (cap %d)" % max_models
        self.ui.tool_preview("consult_models", prompt[:800])
        if not self._approve(summary + "?"):
            return ToolResult(False, "permission denied by user")
        result = self.model_consultant(prompt.strip(), role=role, max_models=max_models)
        return ToolResult(True, self._truncate(str(result)))

    # -- audio ------------------------------------------------------------- #
    @staticmethod
    def _float_arg(args, key: str, default: float,
                   low: float, high: float) -> float:
        try:
            val = float(args.get(key, default))
        except (TypeError, ValueError):
            raise ToolError("%s must be a number" % key)
        if val < low or val > high:
            raise ToolError("%s must be between %s and %s" % (key, low, high))
        return val

    @staticmethod
    def _int_arg(args, key: str, default: int,
                 allowed: Optional[set] = None) -> int:
        try:
            val = int(args.get(key, default))
        except (TypeError, ValueError):
            raise ToolError("%s must be an integer" % key)
        if allowed is not None and val not in allowed:
            raise ToolError("%s must be one of: %s" % (
                key, ", ".join(str(x) for x in sorted(allowed))))
        return val

    def _derive_audio_output(self, source_full: str, profile: str) -> str:
        base, _ext = os.path.splitext(source_full)
        candidate = "%s_%s_master.wav" % (base, profile)
        if not os.path.exists(candidate):
            return candidate
        for idx in range(2, 1000):
            candidate = "%s_%s_master_%d.wav" % (base, profile, idx)
            if not os.path.exists(candidate):
                return candidate
        raise ToolError("could not find an unused output filename")

    def _probe_audio(self, ffprobe: str, source_full: str, timeout: float) -> dict:
        cmd = [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", source_full,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise ToolError("ffprobe failed: " + (proc.stderr or proc.stdout)[:500])
        try:
            data = json.loads(proc.stdout or "{}")
        except ValueError as exc:
            raise ToolError("ffprobe returned invalid JSON: %s" % exc)
        streams = [s for s in data.get("streams", [])
                   if s.get("codec_type") == "audio"]
        if not streams:
            raise ToolError("no audio stream found")
        return {"format": data.get("format", {}), "stream": streams[0]}

    @staticmethod
    def _loudnorm_json(stderr: str) -> Optional[dict]:
        matches = re.findall(r"\{[\s\S]*?\"target_offset\"[\s\S]*?\}", stderr or "")
        for raw in reversed(matches):
            try:
                return json.loads(raw)
            except ValueError:
                continue
        return None

    def _measure_loudnorm(self, ffmpeg: str, source_full: str, target_lufs: float,
                          true_peak: float, lra: float, timeout: float) -> dict:
        filt = "loudnorm=I=%s:TP=%s:LRA=%s:print_format=json" % (
            target_lufs, true_peak, lra)
        cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", source_full,
               "-af", filt, "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise ToolError("ffmpeg loudness analysis failed: " + proc.stderr[:700])
        measured = self._loudnorm_json(proc.stderr)
        if not measured:
            raise ToolError("could not parse ffmpeg loudnorm analysis")
        return measured

    @staticmethod
    def _loudnorm_filter(target_lufs: float, true_peak: float, lra: float,
                         measured: Optional[dict]) -> str:
        base = "loudnorm=I=%s:TP=%s:LRA=%s" % (target_lufs, true_peak, lra)
        if not measured:
            return base + ":print_format=json"
        keys = {
            "measured_I": "input_i",
            "measured_TP": "input_tp",
            "measured_LRA": "input_lra",
            "measured_thresh": "input_thresh",
            "offset": "target_offset",
        }
        parts = [base]
        for out_key, in_key in keys.items():
            if in_key in measured:
                parts.append("%s=%s" % (out_key, measured[in_key]))
        parts.append("linear=true")
        parts.append("print_format=json")
        return ":".join(parts)

    def _master_audio(self, args) -> ToolResult:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            return ToolResult(
                False,
                "ffmpeg and ffprobe are required for master_audio. "
                "Install them first, e.g. sudo apt install ffmpeg",
            )

        source_full = self._resolve(args.get("path", ""))
        if not os.path.isfile(source_full):
            return ToolResult(False, "no such audio file: %s" % args.get("path"))

        profile = str(args.get("profile") or "tunecore").strip().lower()
        if profile not in _AUDIO_PROFILES:
            return ToolResult(False, "unknown profile '%s'; valid profiles: %s" % (
                profile, ", ".join(sorted(_AUDIO_PROFILES))))
        defaults = _AUDIO_PROFILES[profile]

        target_lufs = self._float_arg(
            args, "target_lufs", defaults["target_lufs"], -30.0, -6.0)
        true_peak = self._float_arg(
            args, "true_peak", defaults["true_peak"], -6.0, -0.1)
        lra = self._float_arg(args, "lra", defaults["lra"], 1.0, 30.0)
        sample_rate = self._int_arg(
            args, "sample_rate", defaults["sample_rate"],
            allowed={44100, 48000})
        bit_depth = self._int_arg(
            args, "bit_depth", defaults["bit_depth"],
            allowed=set(_PCM_CODECS))
        channels = self._int_arg(
            args, "channels", defaults["channels"], allowed={1, 2})
        codec = _PCM_CODECS[bit_depth]
        timeout = max(float(args.get("timeout", 0) or 0), self.s.bash_timeout, 600.0)

        raw_output = args.get("output")
        output_full = (self._resolve(str(raw_output)) if raw_output
                       else self._derive_audio_output(source_full, profile))
        if not output_full.lower().endswith(".wav"):
            return ToolResult(False, "master_audio output must be a .wav file")
        if os.path.exists(output_full):
            return ToolResult(False, "output already exists: %s" % self._rel(output_full))

        try:
            info = self._probe_audio(ffprobe, source_full, min(timeout, 60.0))
        except (subprocess.TimeoutExpired, ToolError) as exc:
            return ToolResult(False, str(exc))

        stream = info["stream"]
        fmt = info["format"]
        duration = fmt.get("duration", "?")
        input_desc = "%s Hz, %s channel(s), %ss" % (
            stream.get("sample_rate", "?"), stream.get("channels", "?"), duration)
        rel_in = self._rel(source_full)
        rel_out = self._rel(output_full)
        preview = (
            "input: %s (%s)\n"
            "output: %s\n"
            "profile: %s, %.1f LUFS, %.1f dBTP, LRA %.1f\n"
            "format: %d Hz, %d channel(s), %d-bit WAV (%s)"
            % (rel_in, input_desc, rel_out, profile, target_lufs, true_peak, lra,
               sample_rate, channels, bit_depth, codec)
        )
        self.ui.tool_preview("master_audio", preview)
        if not self._approve("create mastered audio file %s?" % rel_out):
            return ToolResult(False, "permission denied by user")

        try:
            os.makedirs(os.path.dirname(output_full) or ".", exist_ok=True)
            measured = self._measure_loudnorm(
                ffmpeg, source_full, target_lufs, true_peak, lra, timeout)
            loudnorm = self._loudnorm_filter(target_lufs, true_peak, lra, measured)
            filters = "%s,aresample=%d:dither_method=triangular" % (
                loudnorm, sample_rate)
            cmd = [
                ffmpeg, "-hide_banner", "-nostats", "-y", "-i", source_full,
                "-map", "0:a:0", "-vn", "-af", filters,
                "-ar", str(sample_rate), "-ac", str(channels),
                "-c:a", codec, output_full,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "audio mastering timed out after %.0fs" % timeout)
        except ToolError as exc:
            return ToolResult(False, str(exc))

        if proc.returncode != 0:
            return ToolResult(False, "ffmpeg mastering failed: " + proc.stderr[:900])

        self._undo_stack.append((output_full, None))
        final = self._loudnorm_json(proc.stderr) or {}
        metrics = []
        for label, key in (("input_i", "input_i"), ("input_tp", "input_tp"),
                           ("output_i", "output_i"), ("output_tp", "output_tp")):
            if key in final:
                metrics.append("%s=%s" % (label, final[key]))
        metric_text = "; ".join(metrics) if metrics else "final loudness logged by ffmpeg"
        return ToolResult(True, self._truncate(
            "mastered audio created: %s\n"
            "source: %s\n"
            "target: %.1f LUFS, %.1f dBTP true peak, LRA %.1f\n"
            "format: %d Hz, %d channel(s), %d-bit WAV\n"
            "%s\n"
            "[undoable: /undo removes the created master]"
            % (rel_out, rel_in, target_lufs, true_peak, lra,
               sample_rate, channels, bit_depth, metric_text)
        ))
