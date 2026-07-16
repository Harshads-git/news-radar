"""
src/pipeline/retry_budget.py
==============================
Per-provider circuit breaker and adaptive concurrency throttling.

Background
----------
When the AI API starts returning rate-limit or server errors, the default
behaviour (4 retries per call, 5 concurrent calls) makes things worse: all
5 slots simultaneously back off and retry, increasing pressure on the already-
stressed API. This module implements two countermeasures:

  1. **Adaptive concurrency** — After N consecutive failures the scorer
     reduces its semaphore size (fewer simultaneous calls = less pressure).
     The concurrency recovers gradually as calls succeed again.

  2. **Circuit breaker** — Three-state FSM (CLOSED → OPEN → HALF_OPEN):
       CLOSED  : normal operation
       OPEN    : all calls skipped / fail-fast (provider is unavailable)
       HALF_OPEN : one probe call allowed to test if provider recovered

State persistence
-----------------
State is written to ``data/retry_budget.jsonl`` — one line per event
(circuit open/close, throttle change). The ``--retry-stats`` CLI command
reads this file to display a history table.

Usage (typical wiring in NewsScorer)
-------------------------------------
    budget = RetryBudget(data_dir, provider_name="openai")

    # Ask for recommended concurrency (respects throttle state)
    semaphore = asyncio.Semaphore(budget.recommended_concurrency)

    # After each call:
    if call_succeeded:
        budget.record_success()
    else:
        budget.record_failure(error_type)

    # Before each call:
    if budget.is_open:
        raise AIProviderError("Circuit breaker open — provider unavailable")

Configuration constants
-----------------------
    FAILURE_RATE_THRESHOLD   = 0.40  # >40% errors → throttle concurrency
    OPEN_CIRCUIT_THRESHOLD   = 5     # 5 consecutive failures → open circuit
    HALF_OPEN_PROBE_DELAY_S  = 60    # seconds before allowing a probe call
    MIN_CONCURRENCY          = 1     # floor for adaptive throttling
    MAX_CONCURRENCY          = 5     # ceiling (default)
    THROTTLE_STEP            = 1     # reduce by 1 on each spike
    RECOVERY_STEP            = 1     # restore by 1 per successful call batch
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAILURE_RATE_THRESHOLD: float = 0.40   # >40% errors triggers throttle
OPEN_CIRCUIT_THRESHOLD: int = 5        # consecutive failures to open circuit
HALF_OPEN_PROBE_DELAY_S: float = 60.0  # seconds before allowing probe in HALF_OPEN
MIN_CONCURRENCY: int = 1
MAX_CONCURRENCY: int = 5
THROTTLE_STEP: int = 1                 # concurrency reduction per spike
RECOVERY_STEP: int = 1                 # concurrency increase per recovery window
WINDOW_SIZE: int = 20                  # rolling window for error-rate calculation

_BUDGET_FILENAME = "retry_budget.jsonl"


# ---------------------------------------------------------------------------
# Circuit state FSM
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "CLOSED"         # normal — calls flow through
    OPEN = "OPEN"             # tripped — calls fail-fast
    HALF_OPEN = "HALF_OPEN"   # testing — one probe allowed


@dataclass
class CircuitEvent:
    """One recorded event in the circuit breaker / throttle history."""
    timestamp: str
    provider: str
    event_type: str       # "throttle_down", "throttle_up", "circuit_open",
    #                       "circuit_close", "probe_success", "probe_failure"
    old_value: object     # previous concurrency / circuit state
    new_value: object     # new concurrency / circuit state
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "event_type": self.event_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitEvent":
        return cls(
            timestamp=d.get("timestamp", ""),
            provider=d.get("provider", ""),
            event_type=d.get("event_type", ""),
            old_value=d.get("old_value"),
            new_value=d.get("new_value"),
            reason=d.get("reason", ""),
        )


# ---------------------------------------------------------------------------
# RetryBudget
# ---------------------------------------------------------------------------


class RetryBudget:
    """
    Per-provider adaptive concurrency + circuit breaker.

    Thread-safe: all state mutations are guarded by a threading.Lock.

    Parameters
    ----------
    data_dir:
        Root data directory. Events are written to
        ``data_dir / "retry_budget.jsonl"``.
    provider_name:
        Identifier for the AI provider (e.g. "openai", "gemini").
    max_concurrency:
        Upper bound for adaptive concurrency. Default: 5.
    """

    def __init__(
        self,
        data_dir: Path,
        provider_name: str = "openai",
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / _BUDGET_FILENAME
        self.provider = provider_name
        self._max_concurrency = max_concurrency
        self._lock = threading.Lock()

        # Rolling window: list of bools (True=success, False=failure)
        self._window: list[bool] = []

        # Circuit breaker state
        self._circuit_state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._half_open_at: Optional[float] = None  # monotonic clock time

        # Adaptive concurrency
        self._current_concurrency: int = max_concurrency

        # Session counters
        self._total_successes: int = 0
        self._total_failures: int = 0

        # Events buffer (flushed to JSONL)
        self._events: list[CircuitEvent] = []

    # ------------------------------------------------------------------
    # Recording outcomes
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful AI call outcome."""
        with self._lock:
            self._total_successes += 1
            self._window.append(True)
            self._window = self._window[-WINDOW_SIZE:]
            self._consecutive_failures = 0

            # Circuit half-open probe succeeded → close the circuit
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._transition_circuit(CircuitState.CLOSED, reason="probe succeeded")

            # Maybe recover concurrency
            self._maybe_recover_concurrency()

    def record_failure(self, error_type: str = "") -> None:
        """Record a failed AI call outcome."""
        with self._lock:
            self._total_failures += 1
            self._window.append(False)
            self._window = self._window[-WINDOW_SIZE:]
            self._consecutive_failures += 1

            # Check if circuit should open
            if (
                self._circuit_state == CircuitState.CLOSED
                and self._consecutive_failures >= OPEN_CIRCUIT_THRESHOLD
            ):
                self._transition_circuit(
                    CircuitState.OPEN,
                    reason=f"{self._consecutive_failures} consecutive failures ({error_type})",
                )
                import time
                self._half_open_at = time.monotonic() + HALF_OPEN_PROBE_DELAY_S
                return

            # Half-open probe failed → back to open
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._transition_circuit(
                    CircuitState.OPEN,
                    reason=f"probe failed ({error_type})",
                )
                import time
                self._half_open_at = time.monotonic() + HALF_OPEN_PROBE_DELAY_S
                return

            # Maybe throttle concurrency on high error rate
            if self._circuit_state == CircuitState.CLOSED:
                self._maybe_throttle_concurrency(error_type)

    # ------------------------------------------------------------------
    # Querying state
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """
        Returns True if calls should be fast-failed (circuit is OPEN).

        Also handles the OPEN → HALF_OPEN transition after the probe delay.
        """
        import time
        with self._lock:
            if self._circuit_state == CircuitState.OPEN:
                if (
                    self._half_open_at is not None
                    and time.monotonic() >= self._half_open_at
                ):
                    # Allow one probe
                    self._transition_circuit(
                        CircuitState.HALF_OPEN,
                        reason=f"probe delay elapsed ({HALF_OPEN_PROBE_DELAY_S}s)",
                    )
                    return False  # let the probe through
                return True
            return False

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit state (CLOSED / OPEN / HALF_OPEN)."""
        with self._lock:
            return self._circuit_state

    @property
    def recommended_concurrency(self) -> int:
        """Current adaptive concurrency limit (1 ≤ value ≤ max_concurrency)."""
        with self._lock:
            return self._current_concurrency

    @property
    def error_rate(self) -> float:
        """
        Error rate within the rolling window (0.0–1.0).
        Returns 0.0 if the window is empty.
        """
        with self._lock:
            if not self._window:
                return 0.0
            return sum(1 for x in self._window if not x) / len(self._window)

    @property
    def session_stats(self) -> dict:
        """Summary of this session's call outcomes."""
        with self._lock:
            total = self._total_successes + self._total_failures
            return {
                "provider": self.provider,
                "circuit_state": self._circuit_state.value,
                "recommended_concurrency": self._current_concurrency,
                "error_rate": round(self.error_rate, 4),
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "consecutive_failures": self._consecutive_failures,
                "window_size": len(self._window),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Append all buffered events to the JSONL file.

        Silently ignores I/O errors so a write failure never aborts a run.
        """
        with self._lock:
            events_to_write = list(self._events)
            self._events.clear()

        if not events_to_write:
            return

        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for event in events_to_write:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def load_history(self, days: int = 30) -> list[dict]:
        """
        Load circuit events from the last N days, newest first.

        Malformed lines are silently skipped.
        """
        if not self._path.exists():
            return []

        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days)).isoformat()

        entries: list[dict] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # timestamp starts with date portion
                    ts = d.get("timestamp", "")
                    if ts[:10] >= cutoff:
                        entries.append(d)
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return []

        return list(reversed(entries))

    def event_summary(self, days: int = 30) -> dict:
        """
        Return counts of each event type over the last N days.

        Returns
        -------
        dict: event_type → count
        """
        history = self.load_history(days=days)
        counts: dict[str, int] = {}
        for row in history:
            et = row.get("event_type", "unknown")
            counts[et] = counts.get(et, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_circuit(self, new_state: CircuitState, reason: str = "") -> None:
        """Record a circuit state transition (caller must hold self._lock)."""
        old_state = self._circuit_state
        self._circuit_state = new_state
        self._events.append(CircuitEvent(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            provider=self.provider,
            event_type=f"circuit_{new_state.value.lower()}",
            old_value=old_state.value,
            new_value=new_state.value,
            reason=reason,
        ))

    def _maybe_throttle_concurrency(self, error_type: str) -> None:
        """Reduce concurrency if error rate exceeds threshold (caller holds lock)."""
        if len(self._window) < max(5, WINDOW_SIZE // 2):
            return  # not enough data yet

        rate = sum(1 for x in self._window if not x) / len(self._window)
        if rate > FAILURE_RATE_THRESHOLD and self._current_concurrency > MIN_CONCURRENCY:
            old = self._current_concurrency
            self._current_concurrency = max(
                MIN_CONCURRENCY, self._current_concurrency - THROTTLE_STEP
            )
            self._events.append(CircuitEvent(
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                provider=self.provider,
                event_type="throttle_down",
                old_value=old,
                new_value=self._current_concurrency,
                reason=f"error rate {rate:.0%} > threshold ({FAILURE_RATE_THRESHOLD:.0%}) [{error_type}]",
            ))

    def _maybe_recover_concurrency(self) -> None:
        """Restore concurrency if recent window shows low error rate (caller holds lock)."""
        if self._current_concurrency >= self._max_concurrency:
            return
        if len(self._window) < max(5, WINDOW_SIZE // 2):
            return

        rate = sum(1 for x in self._window if not x) / len(self._window)
        if rate < FAILURE_RATE_THRESHOLD / 2:
            old = self._current_concurrency
            self._current_concurrency = min(
                self._max_concurrency, self._current_concurrency + RECOVERY_STEP
            )
            self._events.append(CircuitEvent(
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                provider=self.provider,
                event_type="throttle_up",
                old_value=old,
                new_value=self._current_concurrency,
                reason=f"error rate {rate:.0%} < recovery threshold ({FAILURE_RATE_THRESHOLD / 2:.0%})",
            ))
