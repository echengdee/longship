from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Sequence

from .base import PolicyError


class SingleFlightInferenceWorker:
    """Run at most one non-cancellable synchronous inference at a time.

    Cancelling an await cannot stop native inference safely. The underlying
    future therefore remains fenced as in-flight until its worker really exits;
    later calls fail instead of accumulating or overlapping GPU/runner work.
    """

    def __init__(
        self,
        infer: Callable[[tuple[float, ...]], Sequence[float]],
        *,
        thread_name: str,
    ) -> None:
        self._infer = infer
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name
        )
        self._state_lock = threading.Lock()
        self._inflight: Future[Sequence[float]] | None = None

    def _clear(self, completed: Future[Sequence[float]]) -> None:
        with self._state_lock:
            if self._inflight is completed:
                self._inflight = None

    async def run(self, observation: tuple[float, ...]) -> Sequence[float]:
        with self._state_lock:
            if self._inflight is not None and not self._inflight.done():
                raise PolicyError("policy runner is still completing a previous call")
            future = self._executor.submit(self._infer, observation)
            self._inflight = future
        future.add_done_callback(self._clear)
        return await asyncio.wrap_future(future)

    def close(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
