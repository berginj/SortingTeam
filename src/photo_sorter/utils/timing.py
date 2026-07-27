from __future__ import annotations

from time import perf_counter


class Timer:
    def __init__(self) -> None:
        self._started = perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return max(0, round((perf_counter() - self._started) * 1000))
