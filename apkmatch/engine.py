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


def _substitute_record(rec: dict, mapping) -> dict:
    """Return a copy of `rec` with every LX-namespace reference
    substituted through `mapping` (a dict-like `get(a) -> b`). Used
    by the two-pass engine: after first-pass convergence, every
    mapped LX ref in A's records can be rewritten to its B-side
    image, giving tier-2 matchers the same view of A and B."""
    def sub(x: str) -> str:
        if not x or not x.startswith("LX/"):
            return x
        return mapping.get(x) or x

    def sub_sig(sig: str) -> str:
        out = []; i = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1: out.append(sig[i:]); break
                ref = sig[i:e+1]
                out.append(sub(ref))
                i = e + 1; continue
            out.append(c); i += 1
        return "".join(out)

    new = dict(rec)
    if rec.get("super"): new["super"] = sub(rec["super"])
    new["impls"] = [sub(x) for x in rec.get("impls", ())]
    new["calls"] = sorted({sub(x) for x in rec.get("calls", ())})
    new["facc"]  = sorted({sub(x) for x in rec.get("facc",  ())})
    new["trefs"] = sorted({sub(x) for x in rec.get("trefs", ())})
    new["anns"]  = [sub(x) for x in rec.get("anns", ())]
    new["sigs"]  = [sub_sig(s) for s in rec.get("sigs", ())]
    new["field_types"] = [sub_sig(t) for t in rec.get("field_types", ())]
    new["call_targets"] = [(sub(c), n) for c, n in rec.get("call_targets", ())]
    new["field_targets"] = [(sub(c), n) for c, n in rec.get("field_targets", ())]
    new["line_refs"] = [(ln, sub(c)) for ln, c in rec.get("line_refs", ())]
    return new
from .validators import (
    DEFAULT_VALIDATORS, MappingReader, Score, aggregate,
)


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
    validator_stats: dict = None


class Engine:
    def __init__(
        self,
        a: InMemoryProject,
        b: InMemoryProject,
        matchers=None,
        validators=None,
        confirmation_threshold: float = 0.6,
        displace_margin: float = 0.1,
        tier3_max_iters: int = 20,
        tier3_min_churn: int = 50,
        tier3_quiet_epochs: int = 2,
        validator_revoke_threshold: float = 0.4,
        two_pass: bool = True,
    ):
        self.a = a; self.b = b
        self.matchers = matchers or DEFAULT_MATCHERS
        self.validators = validators if validators is not None else DEFAULT_VALIDATORS
        self.mapping = MutableMapping(
            confirmation_threshold=confirmation_threshold,
            displace_margin=displace_margin,
        )
        self.tier3_max_iters = tier3_max_iters
        self.tier3_min_churn = tier3_min_churn
        self.tier3_quiet_epochs = tier3_quiet_epochs
        self.validator_revoke_threshold = validator_revoke_threshold
        self.two_pass = two_pass
        self.stats: dict[str, dict] = defaultdict(
            lambda: {"proposed": 0, "added": 0, "displaced": 0,
                     "updated": 0, "rejected": 0, "locked-blocked": 0})
        # Dep graph: which mapping keys did (validator_id, pair) consult?
        # Lets us re-score only the pairs whose dependencies have moved.
        self._dep_graph: dict[tuple[str, str, str], frozenset] = {}
        self._validator_stats: dict[str, dict] = defaultdict(
            lambda: {"scored": 0, "boosted": 0, "kept": 0, "revoked": 0,
                     "provisional": 0, "veto": 0})

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

    def _score_pair(self, a_cid: str, b_cid: str) -> tuple[float, bool, list[Score]]:
        """Run all validators on (a, b). Returns (confidence, provisional,
        per-validator scores). Records dep tuples for reactive
        re-evaluation."""
        cand = Candidate(a_cid, b_cid, 0.0, "validator", ())
        scores: list[Score] = []
        for v in self.validators:
            reader = MappingReader(self.mapping)
            s = v.score(cand, self.a, self.b, reader)
            self._dep_graph[(v.id, a_cid, b_cid)] = frozenset(reader.deps)
            scores.append(s)
            st = self._validator_stats[v.id]
            st["scored"] += 1
            if s.provisional: st["provisional"] += 1
            if s.value <= 1: st["veto"] += 1
        conf, prov = aggregate(scores)
        return conf, prov, scores

    def _validate_all(self, label: str) -> None:
        """Score every currently-committed pair.

        Conservative policy:
          * Revoke ONLY on hard veto (an aggregate confidence of 0.0
            from any validator's score=1) AND not locked. This avoids
            ping-ponging tier-3 pairs whose validators are still
            provisional.
          * Confidence is boosted toward the validator aggregate only
            when the validator was NOT provisional — provisional
            scores are advisory; the engine should not let them
            artificially raise confidence above a non-provisional
            matcher's reading.
        """
        if not self.validators:
            return
        t0 = time.time()
        pairs = list(self.mapping)
        boosted = 0; kept = 0; revoked = 0; provisional = 0
        for a, b in pairs:
            cur_conf = self.mapping.confidence(a) or 0.0
            conf, prov, _ = self._score_pair(a, b)
            if prov:
                provisional += 1
            if conf == 0.0 and not self.mapping.is_locked(a):
                # Hard veto from at least one validator. Revoke and
                # blacklist so the same pair isn't re-proposed in
                # later iterations.
                self.mapping._a2b.pop(a, None)
                self.mapping._b2a.pop(b, None)
                self.mapping._conf.pop(a, None)
                self.mapping._matchers.pop((a, b), None)
                self.mapping.mark_negative(a, b)
                revoked += 1
                continue
            if not prov and conf > cur_conf:
                self.mapping._conf[a] = conf
                boosted += 1
                # Promote to locked when validators uniformly agree
                # at high confidence — these pairs should be immune
                # to displacement and re-evaluation.
                if conf >= 0.85:
                    self.mapping._locked.add(a)
            else:
                kept += 1
        _log(f"  validators[{label}]: scored={len(pairs)} boosted={boosted} "
             f"kept={kept} revoked={revoked} provisional={provisional}  "
             f"({time.time()-t0:.1f}s)")

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

        # Validator pass between tier 2 and tier 3 — revoke pairs the
        # validators flat-out disagree with, boost confidence on pairs
        # they corroborate. Locked tier-1 anchors are immune.
        if self.validators:
            _log(f"=== validator pass (after tier 2) ===")
            self._validate_all("post-tier2")


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
                # After each tier-3 sweep, re-run validators: refs that
                # were unresolvable last epoch may now resolve, so
                # SignatureRefValidator can upgrade or veto pairs.
                if self.validators:
                    self._validate_all(f"post-tier3-iter{it+1}")
                if total_churn < self.tier3_min_churn:
                    quiet += 1
                    if quiet >= self.tier3_quiet_epochs:
                        converged = True; break
                else:
                    quiet = 0
        else:
            converged = True

        # Final validator pass for confidence calibration.
        if self.validators:
            _log(f"=== validator pass (final) ===")
            self._validate_all("final")

        # Neighbour-consistency cleanup, iterated. Revoking pairs can
        # remove evidence that supported other pairs, so re-check.
        # First cleanup round, then re-run tier-3 matchers to give
        # revoked slots a chance at a correct match.
        for cleanup_round in range(1):
            _log(f"=== neighbour-consistency cleanup round {cleanup_round+1} ===")
            revoked = 0
            for a, b in list(self.mapping):
                if self.mapping.is_locked(a):
                    continue
                a_nbs = list(self.a.neighbours(a))
                mapped_nbs = [self.mapping.get(n) for n in a_nbs
                              if self.mapping.get(n) is not None]
                if len(mapped_nbs) < 3:
                    continue
                b_nbs = set(self.b.neighbours(b))
                hits = sum(1 for x in mapped_nbs if x in b_nbs)
                if hits == 0:
                    self.mapping._a2b.pop(a, None)
                    self.mapping._b2a.pop(b, None)
                    self.mapping._conf.pop(a, None)
                    self.mapping._matchers.pop((a, b), None)
                    self.mapping.mark_negative(a, b)
                    revoked += 1
            _log(f"  round {cleanup_round+1}: revoked {revoked} pairs")
            if revoked == 0:
                break

        # Second tier-3 pass after cleanup: the revoked pairs left
        # gaps that tier-3 matchers might fill correctly now that
        # noise has been removed. Short-circuit if no churn.
        if tiers.get(3):
            _log(f"=== second tier-3 sweep (post-cleanup) ===")
            for it in range(3):
                _log(f"--- post-cleanup iter {it+1} ---")
                total_churn = 0
                for m in tiers[3]:
                    churn = self._run_matcher(m, mapping_arg=True)
                    total_churn += (churn.confirmed_added
                                    + churn.confirmed_removed
                                    + churn.confirmed_changed)
                if self.validators:
                    self._validate_all(f"post-cleanup-iter{it+1}")
                if total_churn < self.tier3_min_churn:
                    break

        # Two-pass: rebuild A with LX refs substituted, re-run
        # substitution-sensitive tier-2 matchers. Single round —
        # multi-round adds marginal coverage but iterates noise.
        second_pass_ids = {
            "stable_refs_ms", "stable_refs_set", "strings_plus_refs",
            "call_targets_ms", "field_targets_ms", "fqn_stable",
        }
        tier2_subst = [m for m in self.matchers
                       if getattr(m, "tier", 2) == 2
                       and getattr(m, "id", "") in second_pass_ids]
        for rnd in range(0 if not self.two_pass else 2):
            if not len(self.mapping):
                break
            _log(f"=== two-pass round {rnd+1}: substituting + re-running tier-2 ===")
            t0 = time.time()
            class _MapView:
                def __init__(self, m): self._m = m
                def get(self, k): return self._m.get(k)
            sub_records = [_substitute_record(r, _MapView(self.mapping))
                           for r in self.a.classes()]
            from .project import InMemoryProject as _IMP
            sub_a = _IMP(sub_records)
            _log(f"  substituted A in {time.time()-t0:.1f}s")
            orig_a = self.a
            self.a = sub_a
            before = len(self.mapping)
            for m in tier2_subst:
                self._run_matcher(m)
            self.a = orig_a
            if self.validators:
                _log(f"=== validator pass (post two-pass round {rnd+1}) ===")
                self._validate_all(f"post-twopass-r{rnd+1}")
            gained = len(self.mapping) - before
            _log(f"  round {rnd+1} gained {gained} pairs")
            if gained < 50:
                break

        # Final cleanup after two-pass: two-pass can introduce weak
        # pairs based on substituted refs that don't satisfy
        # neighbour consistency in the original view. Iterate twice
        # (revoking a pair removes evidence supporting other weak
        # pairs); empirically 2 rounds picks up another small lift
        # without the over-revocation that 3+ rounds causes.
        for cleanup_round in range(1):
            _log(f"=== neighbour-consistency cleanup post-twopass r{cleanup_round+1} ===")
            revoked = 0
            for a, b in list(self.mapping):
                if self.mapping.is_locked(a):
                    continue
                a_nbs = list(self.a.neighbours(a))
                mapped_nbs = [self.mapping.get(n) for n in a_nbs
                              if self.mapping.get(n) is not None]
                if len(mapped_nbs) < 3:
                    continue
                b_nbs = set(self.b.neighbours(b))
                hits = sum(1 for x in mapped_nbs if x in b_nbs)
                if hits == 0:
                    self.mapping._a2b.pop(a, None)
                    self.mapping._b2a.pop(b, None)
                    self.mapping._conf.pop(a, None)
                    self.mapping._matchers.pop((a, b), None)
                    self.mapping.mark_negative(a, b)
                    revoked += 1
            _log(f"  r{cleanup_round+1}: revoked {revoked} pairs")
            if revoked == 0:
                break

        # Third tier-3 sweep after final cleanup.
        if tiers.get(3):
            _log(f"=== final tier-3 sweep (after all cleanups) ===")
            for it in range(3):
                _log(f"--- final iter {it+1} ---")
                total_churn = 0
                for m in tiers[3]:
                    churn = self._run_matcher(m, mapping_arg=True)
                    total_churn += (churn.confirmed_added
                                    + churn.confirmed_removed
                                    + churn.confirmed_changed)
                if self.validators:
                    self._validate_all(f"final-tier3-iter{it+1}")
                if total_churn < self.tier3_min_churn:
                    break

        return RunResult(
            epochs=self.mapping.epoch,
            converged=converged,
            matcher_stats=dict(self.stats),
            final_size=len(self.mapping),
            a_total=len(self.a),
            b_total=len(self.b),
            validator_stats=dict(self._validator_stats),
        )
