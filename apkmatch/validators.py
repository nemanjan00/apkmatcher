"""CandidateValidators — rate a candidate, declare what they consulted.

Each call returns a `Score` with:

  * value        : an integer 1..10 ALWAYS based on what was knowable.
                   Never used as "I don't know"; that's what
                   `provisional` is for.
  * provisional  : True iff the score would likely change given more
                   data. Drives re-evaluation, NOT the score itself.
  * deps         : the set of mapping keys (ClassIds) the validator
                   consulted. The engine builds a reverse index so it
                   can find which (validator, candidate) results
                   become dirty when a given mapping entry changes.

Validators never read the mapping directly; they read it through a
`MappingReader` wrapper that records each lookup as a dep. That's the
ONLY thing they have to do to participate in the reactive
invalidation engine.

Score scale:
   1  = hard veto (kills the candidate, any number of other 10s lose)
   2-4 = negative
   5  = no opinion, but not deferred — "I looked, it's a wash"
   6-9 = positive
  10  = decisive positive
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .matchers import Candidate
from .project import InMemoryProject

SCORE_VETO = 1
SCORE_NEUTRAL = 5
SCORE_DECISIVE = 10


@dataclass
class Score:
    value: int
    provisional: bool = False
    deps: tuple[str, ...] = ()


class MappingReader:
    """Wraps a MappingView, records every lookup."""

    def __init__(self, mapping):
        self._mapping = mapping
        self._deps: set[str] = set()

    def get(self, a: str) -> Optional[str]:
        self._deps.add(a)
        return self._mapping.get(a)

    def inverse(self, b: str) -> Optional[str]:
        # tag with a sentinel; engine knows inverse deps via the
        # B-side mapping key — but for simplicity we treat any
        # lookup as a dep on A keys only. (Validators rarely need
        # inverse.)
        return self._mapping.inverse(b)

    @property
    def deps(self) -> tuple[str, ...]:
        return tuple(sorted(self._deps))


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

class ShapeValidator:
    """Coarse compatibility. Hard-vetoes mismatched class KIND."""
    id = "shape"

    def score(self, c: Candidate, a: InMemoryProject, b: InMemoryProject,
              reader: MappingReader) -> Score:
        ra = a.get(c.a); rb = b.get(c.b)
        if not ra or not rb:
            return Score(SCORE_VETO)
        ma = set(ra["mods"]); mb = set(rb["mods"])
        for kind in ("interface", "enum", "annotation"):
            if (kind in ma) != (kind in mb):
                return Score(SCORE_VETO)
        def delta(x, y): return abs(x - y) / max(1, max(x, y))
        d = max(delta(ra["nf"], rb["nf"]), delta(ra["nm"], rb["nm"]))
        if d > 0.75: v = 2
        elif d > 0.5:  v = 3
        elif d > 0.25: v = 6
        elif d > 0.1:  v = 8
        else:          v = SCORE_DECISIVE
        return Score(v)  # never provisional — purely structural


class SignatureRefValidator:
    """Substitute matched LX refs into A's signatures and see how many
    appear on B's class.

    Always produces a numeric answer based on what's currently
    resolvable. Sets `provisional=True` when fewer than `min_resolved`
    of the LX refs encountered were resolvable through the mapping.
    """
    id = "signature_refs"

    def __init__(self, min_resolved_ratio: float = 0.5):
        self.min_resolved_ratio = min_resolved_ratio

    def _substitute(self, sig: str, reader: MappingReader) -> tuple[str, int, int]:
        out = []
        i = 0; n_lx = 0; n_res = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1:
                    out.append(sig[i:]); break
                ref = sig[i:e+1]
                if ref.startswith("LX/"):
                    n_lx += 1
                    mapped = reader.get(ref)
                    if mapped is not None:
                        n_res += 1
                        out.append(mapped)
                    else:
                        out.append(ref)  # leave as-is when unresolved
                else:
                    out.append(ref)
                i = e + 1
                continue
            out.append(c); i += 1
        return "".join(out), n_lx, n_res

    def score(self, c: Candidate, a: InMemoryProject, b: InMemoryProject,
              reader: MappingReader) -> Score:
        ra = a.get(c.a); rb = b.get(c.b)
        if not ra or not rb or not ra["sigs"] or not rb["sigs"]:
            return Score(SCORE_NEUTRAL, provisional=False)
        b_sig_set = set(rb["sigs"])
        total_lx = 0; total_res = 0
        hits = 0
        for sig in ra["sigs"]:
            sub, n_lx, n_res = self._substitute(sig, reader)
            total_lx += n_lx; total_res += n_res
            if sub in b_sig_set:
                hits += 1
        # base score from hit rate over A's signatures
        rate = hits / len(ra["sigs"])
        if rate >= 0.95: v = SCORE_DECISIVE
        elif rate >= 0.8: v = 8
        elif rate >= 0.5: v = 6
        elif rate >= 0.25: v = 4
        elif rate >  0:   v = 3
        else:             v = 2
        # provisional iff substantial unresolved LX content
        prov = (total_lx > 0 and total_res < total_lx * self.min_resolved_ratio)
        return Score(v, provisional=prov, deps=reader.deps)


class NeighbourConsistencyValidator:
    """Fraction of A's matched outgoing neighbours that land inside
    B's neighbour set. Provisional when A has few matched neighbours."""
    id = "neighbour_consistency"

    def __init__(self, min_matched: int = 3):
        self.min_matched = min_matched

    def score(self, c: Candidate, a: InMemoryProject, b: InMemoryProject,
              reader: MappingReader) -> Score:
        b_nbs = set(b.neighbours(c.b))
        matched_nbs: list[str] = []
        for nb in a.neighbours(c.a):
            mb = reader.get(nb)
            if mb is not None:
                matched_nbs.append(mb)
        if not matched_nbs:
            return Score(SCORE_NEUTRAL, provisional=True, deps=reader.deps)
        hit = sum(1 for x in matched_nbs if x in b_nbs)
        ratio = hit / len(matched_nbs)
        if ratio >= 0.9:   v = SCORE_DECISIVE
        elif ratio >= 0.7: v = 8
        elif ratio >= 0.4: v = 6
        elif ratio >= 0.2: v = 3
        else:              v = 2
        prov = len(matched_nbs) < self.min_matched
        return Score(v, provisional=prov, deps=reader.deps)


class MethodPairCountValidator:
    """For each confirmed class pair, count how many methods pair
    exactly by substituted signature. If many → score high
    (additional confirmation). If zero with non-trivial method sets
    → score low.

    Provisional when fewer than `min_sigs_to_judge` LX-containing
    method signatures could be resolved through the mapping.
    """
    id = "method_pair_count"

    def __init__(self, min_methods: int = 2, min_pairs_to_pass: int = 2):
        self.min_methods = min_methods
        self.min_pairs_to_pass = min_pairs_to_pass

    @staticmethod
    def _sub_sig(sig: str, reader) -> tuple[str, int]:
        """Returns (substituted_sig, unresolved_count)."""
        out = []; i = 0; unresolved = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1: out.append(sig[i:]); break
                ref = sig[i:e+1]
                if ref.startswith("LX/"):
                    m = reader.get(ref)
                    if m is None:
                        unresolved += 1
                        out.append(ref)
                    else:
                        out.append(m)
                else:
                    out.append(ref)
                i = e + 1; continue
            out.append(c); i += 1
        return "".join(out), unresolved

    def score(self, c: Candidate, a: InMemoryProject, b: InMemoryProject,
              reader: MappingReader) -> Score:
        ra = a.get(c.a); rb = b.get(c.b)
        if not ra or not rb:
            return Score(SCORE_NEUTRAL, deps=reader.deps)
        ma_list = ra.get("methods", ())
        mb_list = rb.get("methods", ())
        if len(ma_list) < self.min_methods or len(mb_list) < self.min_methods:
            return Score(SCORE_NEUTRAL, deps=reader.deps)
        b_sigs = {mb["sig"] for mb in mb_list}
        pairs = 0
        total_unresolved = 0
        for ma in ma_list:
            sub_sig, unresolved = self._sub_sig(ma["sig"], reader)
            total_unresolved += unresolved
            if sub_sig in b_sigs:
                pairs += 1
        provisional = total_unresolved > 0 and pairs < self.min_pairs_to_pass
        rate = pairs / len(ma_list)
        if rate >= 0.8: v = SCORE_DECISIVE
        elif rate >= 0.5: v = 8
        elif rate >= 0.25: v = 6
        elif pairs >= self.min_pairs_to_pass: v = SCORE_NEUTRAL
        elif pairs == 0: v = 2
        else: v = 4
        return Score(v, provisional=provisional, deps=reader.deps)


DEFAULT_VALIDATORS = [
    ShapeValidator(),
    SignatureRefValidator(),
    NeighbourConsistencyValidator(),
    MethodPairCountValidator(),
]


def aggregate(scores: list[Score]) -> tuple[float, bool]:
    """Returns (confidence in [0,1], provisional flag)."""
    if not scores:
        return 0.0, True
    if any(s.value <= SCORE_VETO for s in scores):
        return 0.0, False
    mean = sum(s.value for s in scores) / len(scores)
    conf = (mean - 1) / 9
    prov = any(s.provisional for s in scores)
    return conf, prov
