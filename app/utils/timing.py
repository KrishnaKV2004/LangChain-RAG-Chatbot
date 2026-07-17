"""Latency measurement helpers used across the pipeline for observability."""

import time
from types import TracebackType
from typing import Optional, Type


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    Example::

        with Timer() as t:
            results = retriever.invoke(query)
        log.info("retrieval_done", latency_ms=t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Duration is recorded even when the block raises, so error paths
        # still get latency metrics in their logs.
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
