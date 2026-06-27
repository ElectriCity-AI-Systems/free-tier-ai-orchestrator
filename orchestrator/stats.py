"""A tiny, persistent model leaderboard.

Across sessions it remembers which free/free-tier models actually succeed and
how fast they respond, then nudges selection toward the proven performers.
Everything is best-effort: any I/O error degrades to "no stats" rather than
failing the run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Tuple


def default_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".ofo", "leaderboard.json")


class Leaderboard:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._data = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            self._data = {}

    def record(self, model_id: str, ok: bool, latency: float = None) -> None:
        with self._lock:
            rec = self._data.setdefault(model_id, {"ok": 0, "fail": 0, "ema_latency": 0.0})
            rec["ok" if ok else "fail"] = rec.get("ok" if ok else "fail", 0) + 1
            if ok and latency is not None:
                prev = rec.get("ema_latency") or latency
                rec["ema_latency"] = round(0.7 * prev + 0.3 * latency, 3)
            rec["last_seen"] = int(time.time())
            self._dirty = True

    def bias(self, model_id: str) -> float:
        """Small score nudge in roughly [-4, +6] from win-rate and speed."""
        rec = self._data.get(model_id)
        if not rec:
            return 0.0
        ok, fail = rec.get("ok", 0), rec.get("fail", 0)
        total = ok + fail
        if total < 2:
            return 0.0
        winrate = ok / total
        confidence = min(total / 10.0, 1.0)        # trust grows with samples
        score = (winrate - 0.5) * 8.0 * confidence  # ~[-4, +4]
        lat = rec.get("ema_latency") or 0.0
        if lat:                                     # reward fast models a touch
            score += max(-1.0, min(2.0, (8.0 - lat) / 4.0))
        return round(score, 3)

    def top(self, n: int = 10) -> List[Tuple[str, dict]]:
        scored = [(mid, rec, self.bias(mid)) for mid, rec in self._data.items()]
        scored.sort(key=lambda t: t[2], reverse=True)
        return [(mid, rec) for mid, rec, _ in scored[:n]]

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, sort_keys=True)
                os.replace(tmp, self.path)
                self._dirty = False
        except OSError:
            pass  # best-effort; never break a run over stats
