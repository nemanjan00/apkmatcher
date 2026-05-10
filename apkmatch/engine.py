"""Engine — drives the matching pipeline.

The engine runs matchers tier by tier:

  tier 1   -> propose candidates, score with validators, commit;
              survivors are lockable.
  tier 2   -> same.
  tier 3   -> iterate until converged. Each iteration may displace
              earlier confirmations (subject to the displacement
              margin and lock set).

Convergence: K consecutive epochs where churn (added + removed +
changed) is below `min_churn`. Forward-progress guarantee: if an
epoch produces neither churn nor deferral reduction, terminate.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .mapping import EpochChurn, MutableMapping
from .matchers import Candidate, DEFAULT_MATCHERS
from .project import InMemoryProject


def _log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


@dataclass
class RunResult:
    epochs: int
    converged: bool
    matcher_stats: dict
    final_size: int
    a_total: int
    b_total: int


class Engine:
    def __init__(
        self,
        a: InMemoryProject,
        b: InMemoryProject,
        matchers=None,
        confirmation_threshold: float = 0.6,
        displace_margin: float = 0.1,
        tier3_max_iters: int = 10,
        tier3_min_churn: int = 50,
        tier3_quiet_epochs: int = 2,
    ):
        self.a = a; self.b = b
        self.matchers = matchers or DEFAULT_MATCHERS
        self.mapping = MutableMapping(
            confirmation_threshold=confirmation_threshold,
            displace_margin=displace_margin,
        )
        self.tier3_max_iters = tier3_max_iters
        self.tier3_min_churn = tier3_min_churn
        self.tier3_quiet_epochs = tier3_quiet_epochs
        self.stats: dict[str, dict] = defaultdict(
            lambda: {"proposed": 0, "added": 0, "displaced": 0,
                     "updated": 0, "rejected": 0, "locked-blocked": 0})

    def _run_matcher(self, m, mapping_arg: bool = False):
        """Run a single matcher, commit results, return EpochChurn."""
        churn = EpochChurn(epoch=self.mapping.epoch + 1)
        s = self.stats[m.id]
        t0 = time.time()
        if mapping_arg:
            it = m.propose(self.a, self.b, self.mapping)
        else:
            it = m.propose(self.a, self.b)
        cands = list(it)
        t_propose = time.time() - t0
        churn.candidates_emitted = len(cands)
        s["proposed"] += len(cands)
        _log(f"  {m.id:22s} proposed={len(cands):>8d}  ({t_propose:.1f}s)")
        cands.sort(key=lambda c: c.confidence, reverse=True)
        before = dict(self.mapping._a2b)  # snapshot for churn
        before_conf = dict(self.mapping._conf)
        for c in cands:
            outcome = self.mapping.propose(
                c.a, c.b, c.confidence, [m.matcher_id if hasattr(m, "matcher_id") else m.id],
                lock=(m.tier == 1),
            )
            s[outcome] = s.get(outcome, 0) + 1
        # Compute churn
        after = self.mapping._a2b
        after_conf = self.mapping._conf
        added = 0; removed = 0; changed = 0; conf_deltas = []
        for k, v in after.items():
            if k not in before:
                added += 1
            elif before[k] != v:
                changed += 1
            d = abs(after_conf.get(k, 0) - before_conf.get(k, 0))
            if d:
                conf_deltas.append(d)
        for k in before:
            if k not in after:
                removed += 1
        churn.confirmed_added = added
        churn.confirmed_removed = removed
        churn.confirmed_changed = changed
        churn.mean_confidence_delta = (sum(conf_deltas) / len(conf_deltas)
                                       if conf_deltas else 0.0)
        self.mapping.commit_epoch(churn)
        t_commit = time.time() - t0 - t_propose
        _log(f"     +added={s.get('added',0)} updated={s.get('updated',0)} "
             f"displaced={s.get('displaced',0)} rejected={s.get('rejected',0)} "
             f"locked-blocked={s.get('locked-blocked',0)}  ({t_commit:.1f}s commit)")
        _log(f"     mapping size now {len(self.mapping)}")
        return churn

    def run(self) -> RunResult:
        # Group matchers by tier
        tiers: dict[int, list] = defaultdict(list)
        for m in self.matchers:
            tiers[getattr(m, "tier", 2)].append(m)

        # Tiers 1 & 2: single pass each
        for tier in sorted(tiers.keys()):
            if tier >= 3:
                continue
            _log(f"=== tier {tier} ({len(tiers[tier])} matchers) ===")
            for m in tiers[tier]:
                self._run_matcher(m)

        # Tier 3: iterate to fixpoint
        converged = False
        if tiers.get(3):
            _log(f"=== tier 3 ({len(tiers[3])} matchers, max {self.tier3_max_iters} iters) ===")
            quiet = 0
            for it in range(self.tier3_max_iters):
                _log(f"--- iter {it+1} ---")
                total_churn = 0
                for m in tiers[3]:
                    churn = self._run_matcher(m, mapping_arg=True)
                    total_churn += (churn.confirmed_added + churn.confirmed_removed
                                    + churn.confirmed_changed)
                if total_churn < self.tier3_min_churn:
                    quiet += 1
                    if quiet >= self.tier3_quiet_epochs:
                        converged = True; break
                else:
                    quiet = 0
        else:
            converged = True

        return RunResult(
            epochs=self.mapping.epoch,
            converged=converged,
            matcher_stats=dict(self.stats),
            final_size=len(self.mapping),
            a_total=len(self.a),
            b_total=len(self.b),
        )
