"""Work that runs beside the request path (DESIGN §4, §6.5): background evaluation and automatic recalibration.

The request path itself (analyse → choose → transform → call → record → evaluate → learn) lives in
``AdaptiveHarness``; this module only adds what must not block it.

- **Background evaluation.** ``generate(..., check=fn, background=True)`` returns the response at once; the check
  and the evidence update run on one worker thread, in submission order. ``flush()`` waits for them.
- **Automatic recalibration** (opt-in, ``CalibrationConfig.auto``). When the model's profile is ``cold`` (never
  calibrated) or ``stale`` (drift, or a model change), a calibration run starts on a background thread.
  - It never spends money unless allowed. A local provider (e.g. Ollama) needs nothing more. A paid one needs
    ``max_cost_per_call_usd > 0`` and enough harness budget; every call is reserved in the budget ledger first.
  - At most one run at a time, and at most one start per ``min_interval_s``, so a failing run cannot loop.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from ._logging import get_logger

if TYPE_CHECKING:
    from .api import AdaptiveHarness
    from .calibration import CalibrationReport

log = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, harness: AdaptiveHarness) -> None:
        self.h = harness
        self._pool: ThreadPoolExecutor | None = None
        self._pending: set[Future[object]] = set()
        self._lock = threading.Lock()
        self._calibration: threading.Thread | None = None
        self._last_start = 0.0
        self.last_report: CalibrationReport | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ background evaluation
    def submit(self, job: Callable[[], object]) -> Future[object]:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harness-eval")
            fut = self._pool.submit(self._safe, job)
            self._pending.add(fut)
            fut.add_done_callback(self._pending.discard)
            return fut

    @staticmethod
    def _safe(job: Callable[[], object]) -> object:
        try:
            return job()
        except Exception as e:  # never let a background job kill the worker
            log.warning("background job failed: %s: %s", type(e).__name__, e)
            return None

    def flush(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        _, not_done = wait(list(self._pending), timeout=timeout)
        thread = self._calibration
        if thread is not None:
            thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        return not not_done

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

    # ------------------------------------------------------------------ automatic recalibration
    def allowed(self) -> tuple[bool, str]:
        """Whether automatic calibration may run for this harness, and why not."""
        cc = self.h.config.calibration
        if not cc.auto:
            return False, "automatic calibration is off"
        if self.h.store is None or self.h.ledger is None:
            return False, "needs persistence"
        if not self.h.adapter.capabilities().local:
            if cc.max_cost_per_call_usd <= 0:
                return False, "paid provider: set calibration.max_cost_per_call_usd > 0"
            if self.h.ledger.remaining_usd() < cc.max_cost_per_call_usd:
                return False, "paid provider: harness budget exhausted"
        return True, ""

    def maybe_recalibrate(self, status: str | None) -> bool:
        """Start a background calibration if the profile needs one and it is allowed. True if one started."""
        if status not in ("cold", "stale"):
            return False
        ok, _ = self.allowed()
        if not ok:
            return False
        with self._lock:
            if self._calibration is not None and self._calibration.is_alive():
                return False
            if (
                time.monotonic() - self._last_start < self.h.config.calibration.min_interval_s
                and self._last_start
            ):
                return False
            self._last_start = time.monotonic()
            self._calibration = threading.Thread(
                target=self._run_calibration, name="harness-calibration", daemon=True
            )
            self._calibration.start()
        log.info("profile is %s: automatic calibration started", status)
        return True

    def _run_calibration(self) -> None:
        from .calibration import Calibrator

        cc = self.h.config.calibration
        try:
            self.last_report = Calibrator(
                self.h,
                max_calls=cc.max_calls,
                max_cost_per_call_usd=cc.max_cost_per_call_usd,
                max_tokens=cc.max_tokens,
            ).run(categories=cc.categories)
            self.last_error = None
        except Exception as e:  # a failed calibration never affects the user's calls
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("automatic calibration failed: %s", self.last_error)
