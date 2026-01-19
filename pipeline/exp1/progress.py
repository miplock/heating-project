# Ten plik zawiera pomocnicze funkcje do paska postepu.
"""Simple progress bar helper for exp1 scripts."""

from __future__ import annotations

import time


def _fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


class ProgressTracker:
    """Track progress with percent, elapsed time, and ETA."""

    def __init__(self, prefix: str, total: int, width: int = 28) -> None:
        self.prefix = prefix
        self.total = max(total, 1)
        self.width = width
        self.start = time.time()

    def update(self, index: int, *, suffix: str = "") -> None:
        """Update the progress bar line.

        Args:
            index: Current item index (1-based).
            suffix: Optional text appended at the end of the line.
        """
        index = min(max(index, 0), self.total)
        filled = int(self.width * index / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = 100.0 * index / self.total
        elapsed = time.time() - self.start
        rate = elapsed / index if index > 0 else 0.0
        eta = (self.total - index) * rate
        parts = [
            f"{self.prefix} [{bar}]",
            f"{pct:5.1f}%",
            f"{index}/{self.total}",
            f"elapsed {_fmt_time(elapsed)}",
            f"eta {_fmt_time(eta)}",
        ]
        if suffix:
            parts.append(suffix)
        msg = " ".join(parts)
        print(msg, end="\r", flush=True)

    def finish(self) -> None:
        """Finish the progress bar with a newline."""
        print()


def print_progress(
    prefix: str,
    index: int,
    total: int,
    *,
    suffix: str = "",
) -> None:
    """Legacy single-line progress helper."""
    tracker = ProgressTracker(prefix, total)
    tracker.update(index, suffix=suffix)


def finish_progress() -> None:
    """Finish the progress bar with a newline."""
    print()
