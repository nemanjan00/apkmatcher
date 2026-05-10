"""Concrete Matcher implementations.

Each matcher is a callable yielding (a_id, b_id, confidence, evidence)
tuples. The engine drives them tier by tier; they don't know the
schedule.

Findings from exploration on Instagram 2026-02-04 vs 2026-03-16:
  * LX/ class names rotate every build — FQN match restricted to
    stable (non-LX) namespaces only.
  * Internal call/sig refs to LX/... rotate with the class names,
    so body-hash and signature matchers under-match.
  * String literals and references to stable (non-LX) types do NOT
    rotate. They are the most discriminating signal.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from .project import InMemoryProject, _stable


@dataclass
class Candidate:
    a: str
    b: str
    confidence: float
    matcher_id: str
    evidence: tuple = ()


def _fp(*parts) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(repr(p).encode()); h.update(b"\x00")
    return h.hexdigest()


def _stable_refs(project: InMemoryProject, cid: str) -> list[str]:
    return project.stable_refs(cid)


# --------------------------------------------------------------------------- #
# Tier 1 — anchors (high precision, low recall on heavily obfuscated apps)
# --------------------------------------------------------------------------- #

class FQNStable:
    """FQN identity within stable namespaces only. Hard-locks the pair."""
    id = "fqn_stable"; tier = 1

    def propose(self, a: InMemoryProject, b: InMemoryProject) -> Iterator[Candidate]:
        for cid in a.ids():
            if _stable(cid) and b.get(cid) is not None:
                yield Candidate(cid, cid, 1.0, self.id, ("fqn",))


class NativeSymbolSet:
    """Classes with native methods, matched by the *set* of native method
    symbol names. JNI symbol names are usually preserved verbatim."""
    id = "native_syms"; tier = 1

    def propose(self, a, b):
        def H(rec):
            syms = sorted(set(rec["native_syms"]))
            return _fp("nv", syms) if syms else None
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(r)
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.95, self.id,
                                ("native_syms", tuple(sorted(set(r["native_syms"])))))


# --------------------------------------------------------------------------- #
# Tier 2 — content fingerprints
# --------------------------------------------------------------------------- #

class UniqueString:
    """A literal that appears in exactly one A class and exactly one B
    class — those two classes are paired."""
    id = "unique_string"; tier = 2

    def propose(self, a, b):
        sA = defaultdict(list); sB = defaultdict(list)
        for r in a.classes():
            for s in set(r["strings"]):
                if len(s) >= 6:
                    sA[s].append(r["id"])
        for r in b.classes():
            for s in set(r["strings"]):
                if len(s) >= 6:
                    sB[s].append(r["id"])
        for s, alst in sA.items():
            if len(alst) != 1: continue
            blst = sB.get(s)
            if blst and len(blst) == 1:
                yield Candidate(alst[0], blst[0], 0.85, self.id, ("unique_str", s))


class StringsPlusStableRefs:
    """Hash of (non-trivial string set) + (stable type refs set)."""
    id = "strings_plus_refs"; tier = 2

    def propose(self, a, b):
        def H(p, cid):
            r = p.get(cid)
            ss = sorted({s for s in r["strings"] if len(s) >= 4})
            refs = sorted(set(_stable_refs(p, cid)))
            if not ss and len(refs) < 3:
                return None
            return _fp("spr", ss, refs)
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(b, r["id"])
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(a, r["id"])
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.8, self.id, ("strings+refs",))


class StableRefsMultiset:
    """Multiset hash of stable type refs (no strings). Catches classes
    with no string literals but a distinctive set of external API calls."""
    id = "stable_refs_ms"; tier = 2

    def propose(self, a, b):
        def H(p, cid):
            refs = _stable_refs(p, cid)
            if len(refs) < 4: return None
            return _fp("srm", sorted(refs))
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(b, r["id"])
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(a, r["id"])
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.7, self.id, ("stable_refs_ms",))


class IdenticalStrings:
    """Tier-1 anchor: identical SET of string literals.

    When two classes have the *same* set of distinct non-trivial strings,
    they are essentially the same class. The signal is strong as long as
    the set is large enough to be unlikely by chance — we require at
    least `min_strings` distinct strings each at least 4 chars, with the
    pooled length above `min_total_len`.
    """
    id = "identical_strings"; tier = 1

    def __init__(self, min_strings: int = 3, min_total_len: int = 30):
        self.min_strings = min_strings
        self.min_total_len = min_total_len

    def propose(self, a, b):
        def H(rec):
            ss = sorted({s for s in rec["strings"] if len(s) >= 4})
            if len(ss) < self.min_strings:
                return None
            if sum(len(s) for s in ss) < self.min_total_len:
                return None
            return _fp("idstr", ss)
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(r)
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.97, self.id,
                                ("identical_strings",))


class StringSetHash:
    """Same as IdenticalStrings but with a lower bar — kept as a
    tier-2 catcher for classes whose string set is small but still
    matches exactly."""
    id = "stringset_hash"; tier = 2

    def propose(self, a, b):
        def H(rec):
            ss = sorted({s for s in rec["strings"] if len(s) >= 4})
            return _fp("ss", ss) if len(ss) >= 2 else None
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(r)
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.65, self.id, ("string_set",))


class LongUniqueString:
    """A *single* long string literal (>= 20 chars) that appears in
    exactly one class on each side. Long strings collide far less than
    short ones — this is essentially zero-false-positive when it fires."""
    id = "long_unique_string"; tier = 2

    def propose(self, a, b):
        sA = defaultdict(list); sB = defaultdict(list)
        for r in a.classes():
            for s in set(r["strings"]):
                if len(s) >= 20:
                    sA[s].append(r["id"])
        for r in b.classes():
            for s in set(r["strings"]):
                if len(s) >= 20:
                    sB[s].append(r["id"])
        for s, alst in sA.items():
            if len(alst) != 1: continue
            blst = sB.get(s)
            if blst and len(blst) == 1:
                yield Candidate(alst[0], blst[0], 0.95, self.id,
                                ("long_unique_str", s[:40]))


class StringPair:
    """Hash of any 2 unique strings (length >= 8) in a class. Two
    independent strings co-occurring in only one class on each side is
    a strong identity signal even when each string alone collides."""
    id = "string_pair"; tier = 2

    def propose(self, a, b):
        def H(rec):
            ss = sorted({s for s in rec["strings"] if len(s) >= 8})
            return _fp("sp", ss) if len(ss) >= 2 else None
        Bidx = defaultdict(list)
        for r in b.classes():
            h = H(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = H(r)
            if not h: continue
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.85, self.id, ("string_pair",))


# --------------------------------------------------------------------------- #
# Tier 3 — structural / propagation (consult mapping)
# --------------------------------------------------------------------------- #

class NeighbourVote:
    """For each unmatched A class, propose the B class that the most of
    A's already-matched outgoing-neighbours' partners reverse-reference.

    Roughly: 'I look like the class my matched friends point at.'

    Optimized: invert by walking the mapping rather than every A class.
    For each (X→X') in the mapping, iterate B's reverse-neighbours of
    X' to find B classes B'. For each A neighbour of X, that A class
    casts a vote for B'. Skip mapped-class buckets whose B-reverse
    neighbour set is too large (noise).
    """
    id = "neighbour_vote"; tier = 3

    def __init__(self, min_votes: int = 4, max_rev_b: int = 200):
        self.min_votes = min_votes
        self.max_rev_b = max_rev_b

    def propose(self, a, b, mapping):
        votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a_x, b_x in mapping:
            rev_b = list(b.reverse_neighbours(b_x))
            if not rev_b or len(rev_b) > self.max_rev_b:
                continue
            for a_voter in a.reverse_neighbours(a_x):
                if mapping.get(a_voter) is not None:
                    continue  # already matched
                voter_votes = votes[a_voter]
                for nb in rev_b:
                    voter_votes[nb] += 1
        for a_cid, vmap in votes.items():
            if not vmap:
                continue
            # top-1 with margin over runner-up
            top_b, top_v = max(vmap.items(), key=lambda kv: kv[1])
            if top_v < self.min_votes:
                continue
            # check unique-ness: require margin
            second = max((v for k, v in vmap.items() if k != top_b), default=0)
            if top_v - second < 2:
                continue
            conf = min(0.55 + 0.03 * (top_v - second), 0.9)
            yield Candidate(a_cid, top_b, conf, self.id,
                            ("neighbour_vote", top_v, second))


DEFAULT_MATCHERS = [
    # ---- Tier 1: anchors, lockable ------------------------------------
    FQNStable(),             # non-LX FQN identity
    NativeSymbolSet(),       # JNI symbol set
    IdenticalStrings(),      # large identical string set
    LongUniqueString(),      # any string >= 20 chars unique on both sides

    # ---- Tier 2: content fingerprints ---------------------------------
    UniqueString(),          # globally unique string (any length >= 6)
    StringPair(),            # 2+ strings >= 8 chars
    StringSetHash(),         # any string-set match
    StringsPlusStableRefs(), # strings + non-LX refs
    StableRefsMultiset(),    # non-LX refs alone

    # ---- Tier 3: propagation, iterated --------------------------------
    NeighbourVote(min_votes=4),
]
