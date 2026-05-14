"""MutableMapping — concrete MappingView with epoch and churn tracking.

The mapping is mutable inside the engine; matchers and validators only
ever see it through the read-only MappingView surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Set


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
                 displace_margin: float = 0.1,
                 a_stable_ids: Optional[set] = None,
                 b_stable_ids: Optional[set] = None):
        self.confirmation_threshold = confirmation_threshold
        self.displace_margin = displace_margin
        # Set of stable (non-LX) class IDs present in each project.
        # Used by the cross-namespace guard: a stable class in A is
        # only allowed to pair with a different namespace in B if its
        # FQN does NOT exist as a stable class in B (i.e., it was
        # rotated into LX/ in B). Same in reverse. Without these
        # sets, the guard treats any stable<->LX pair as nonsense,
        # which is correct for same-version compares but blocks
        # long-range matching (e.g. v226→v415, where many com/insta-
        # gram/* classes ARE the correct partner for v415's LX/* ids).
        self._a_stable_ids: set = a_stable_ids or set()
        self._b_stable_ids: set = b_stable_ids or set()
        self._a2b: dict[str, str] = {}
        self._b2a: dict[str, str] = {}
        self._conf: dict[str, float] = {}
        self._locked: set[str] = set()
        self._matchers: dict[tuple[str, str], list[str]] = {}
        # Hard negatives: pairs the engine has revoked via veto are
        # NOT re-acceptable unless a tier-1 matcher (lockable)
        # proposes them. Stops the revoke-then-re-add oscillation.
        self._negative: set[tuple[str, str]] = set()
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
        'rejected' / 'locked-blocked' / 'negative' / 'cross-ns'."""

        # Cross-namespace guard. Stable (non-LX) classes mostly
        # round-trip by FQN, so a stable<->stable pair must have
        # equal FQNs. For mixed stable<->LX pairs, we used to reject
        # them outright (which is correct for v415<->v416 where both
        # builds share namespacing conventions). For long-range
        # matches like v226->v415 the same class can be stable in
        # one build and rotated into LX/ in the other; that's the
        # legit case the guard must allow. Permit the cross-ns pair
        # only when the stable side's FQN does NOT exist as a stable
        # class in the OTHER project — i.e. it was rotated.
        a_stable = not a.startswith("LX/")
        b_stable = not b.startswith("LX/")
        if a_stable and b_stable:
            if a != b:
                return "cross-ns"
        elif a_stable and not b_stable:
            # A is stable, B is LX. Allowed iff A's FQN is not
            # present as a stable class in B (i.e. it was rotated).
            if a in self._b_stable_ids:
                return "cross-ns"
        elif b_stable and not a_stable:
            # B is stable, A is LX. Allowed iff B's FQN is not
            # present as a stable class in A.
            if b in self._a_stable_ids:
                return "cross-ns"

        cur_b = self._a2b.get(a)
        cur_b_owner = self._b2a.get(b)

        # Tier-1 lockable matchers can override negative cache;
        # everyone else is bound by it.
        if (a, b) in self._negative and not lock:
            return "negative"
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

    def mark_negative(self, a: str, b: str) -> None:
        self._negative.add((a, b))

    def matchers_for(self, a: str, b: str) -> list[str]:
        return list(self._matchers.get((a, b), []))

    def commit_epoch(self, churn: EpochChurn) -> None:
        self._epoch += 1
        churn.epoch = self._epoch
        self._churn = churn
