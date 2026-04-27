"""Tiny progress-logging helpers.

Used to print periodic status updates to the console without spamming
(throttled by wall-clock seconds). Pure stdlib; works in any terminal,
including IDE consoles that do not handle carriage-return repaints well.
"""

from __future__ import annotations

import sys
import time


class ProgressLogger:
    """Throttled status printer.

    Call ``log(message)`` as often as you like; output is suppressed until
    ``min_interval`` seconds have passed since the last printed line. The
    very first call and any call with ``force=True`` always prints.
    """

    def __init__(self, min_interval: float = 1.0, stream=sys.stdout, prefix: str = ""):
        self.min_interval = min_interval
        self.stream = stream
        self.prefix = prefix
        self._t_start = time.time()
        self._t_last = 0.0  # so the first .log() always fires

    def reset(self) -> None:
        self._t_start = time.time()
        self._t_last = 0.0

    def elapsed(self) -> float:
        return time.time() - self._t_start

    def log(self, message: str, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._t_last) < self.min_interval:
            return
        self._t_last = now
        line = f"{self.prefix}{message}"
        print(line, file=self.stream, flush=True)


def fmt_eta(elapsed: float, fraction_done: float) -> str:
    """Return a human-readable ETA string given elapsed time and fraction done."""
    if fraction_done <= 0:
        return "?"
    total = elapsed / fraction_done
    eta = max(total - elapsed, 0.0)
    if eta < 60:
        return f"{eta:4.1f}s"
    if eta < 3600:
        return f"{eta / 60:4.1f}m"
    return f"{eta / 3600:4.1f}h"
