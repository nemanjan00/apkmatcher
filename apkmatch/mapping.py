"""MutableMapping — concrete MappingView with epoch and churn tracking.

The mapping is mutable inside the engine; matchers and validators only
ever see it through the read-only MappingView surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class EpochChurn:
    epoch: int
    confirmed_added: int = 0
    confirmed_removed: int = 0
    confirmed_changed: int = 0
    candidates_emitted: int = 0
    mean_confidence_delta: float = 0.0
    pending_deferrals: int = 0
    deferrals_resolved: int = 0


class MutableMapping:
    """The engine writes here; the view methods satisfy MappingView."""

    def __init__(self, confirmation_threshold: float = 0.6,
                 displace_margin: float = 0.1):
        self.confirmation_threshold = confirmation_threshold
        self.displace_margin = displace_margin
        self._a2b: dict[str, str] = {}
        self._b2a: dict[str, str] = {}
        self._conf: dict[str, float] = {}
        self._locked: set[str] = set()
        self._matchers: dict[tuple[str, str], list[str]] = {}
        self._epoch = 0
        self._churn = EpochChurn(epoch=0)

    # --- MappingView ---------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def get(self, a: str) -> Optional[str]:
        return self._a2b.get(a)

    def inverse(self, b: str) -> Optional[str]:
        return self._b2a.get(b)

    def confidence(self, a: str) -> Optional[float]:
        return self._conf.get(a)

    def is_confirmed(self, a: str, b: str) -> bool:
        return self._a2b.get(a) == b and self._conf.get(a, 0.0) >= self.confirmation_threshold

    def is_locked(self, a: str) -> bool:
        return a in self._locked

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._a2b.items())

    def __len__(self) -> int:
        return len(self._a2b)

    @property
    def churn(self) -> EpochChurn:
        return self._churn

    # --- mutation (engine only) ----------------------------------------

    def propose(self, a: str, b: str, confidence: float,
                matcher_ids: list[str], lock: bool = False) -> str:
        """Returns the outcome: 'added' / 'updated' / 'displaced' /
        'rejected' / 'locked-blocked'."""
        cur_b = self._a2b.get(a)
        cur_b_owner = self._b2a.get(b)

        if a in self._locked and cur_b != b:
            return "locked-blocked"
        if cur_b_owner and cur_b_owner != a and cur_b_owner in self._locked:
            return "locked-blocked"

        if cur_b == b:
            old = self._conf.get(a, 0.0)
            if confidence > old:
                self._conf[a] = confidence
                self._matchers.setdefault((a, b), []).extend(matcher_ids)
            if lock:
                self._locked.add(a)
            return "updated"

        cur_conf_a = self._conf.get(a, 0.0) if cur_b else 0.0
        cur_conf_b = self._conf.get(cur_b_owner, 0.0) if cur_b_owner else 0.0
        if confidence < max(cur_conf_a, cur_conf_b) + self.displace_margin:
            return "rejected"

        if cur_b is not None:
            self._b2a.pop(cur_b, None)
            self._conf.pop(a, None)
            self._matchers.pop((a, cur_b), None)
        if cur_b_owner is not None and cur_b_owner != a:
            self._a2b.pop(cur_b_owner, None)
            self._conf.pop(cur_b_owner, None)
            self._matchers.pop((cur_b_owner, b), None)

        self._a2b[a] = b
        self._b2a[b] = a
        self._conf[a] = confidence
        self._matchers[(a, b)] = list(matcher_ids)
        if lock:
            self._locked.add(a)
        return "added" if cur_b is None and cur_b_owner is None else "displaced"

    def matchers_for(self, a: str, b: str) -> list[str]:
        return list(self._matchers.get((a, b), []))

    def commit_epoch(self, churn: EpochChurn) -> None:
        self._epoch += 1
        churn.epoch = self._epoch
        self._churn = churn
