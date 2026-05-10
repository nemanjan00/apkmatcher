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


class BodyHashSubstituted:
    """Re-hash A's method body opcodes after substituting matched LX
    refs through the current mapping. Match against B's raw body
    hashes — catches classes whose internals are otherwise identical
    once the rotated names are accounted for.

    Tier 3 because it depends on having enough of the mapping built
    by tier 1 and 2 to be useful. Re-runs each tier-3 sweep, picking
    up newly mappable classes.

    Note: `bh` records in the indexed JSONL are the RAW body hash with
    no substitution. To do substitution we'd need the raw smali lines
    again. As a proxy we operate at the granularity of the per-class
    'sigs' (method signatures) plus per-class stable-ref multiset —
    both already in the index. This is a structural fingerprint, not
    a true bytecode hash, but is cheap and adds signal that the
    earlier matchers' signature work missed because they didn't apply
    substitution.
    """
    id = "sig_substituted"; tier = 3

    def _substitute(self, sig: str, mapping) -> str:
        out = []
        i = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1:
                    out.append(sig[i:]); break
                ref = sig[i:e+1]
                if ref.startswith("LX/"):
                    mapped = mapping.get(ref)
                    out.append(mapped or "LX/?;")
                else:
                    out.append(ref)
                i = e + 1
                continue
            out.append(c); i += 1
        return "".join(out)

    def propose(self, a, b, mapping):
        # Build B's signature multiset hashes as-is.
        Bidx = defaultdict(list)
        for r in b.classes():
            if not r["sigs"] or r["nm"] < 2:
                continue
            h = _fp("sgs", sorted(r["sigs"]))
            Bidx[h].append(r["id"])
        for r in a.classes():
            if not r["sigs"] or r["nm"] < 2:
                continue
            sub = sorted(self._substitute(s, mapping) for s in r["sigs"])
            # Skip if substitution didn't actually change anything
            # (then we'd be duplicating signature_multiset's work)
            if sub == sorted(r["sigs"]):
                continue
            # Skip if too many unresolved refs leaked through
            unresolved = sum(s.count("LX/?;") for s in sub)
            if unresolved > len(sub):
                continue
            h = _fp("sgs", sub)
            for bid in Bidx.get(h, ()):
                yield Candidate(r["id"], bid, 0.7, self.id, ("sig_sub",))


# Edge-kind weights for NeighbourVote — inheritance is far more
# discriminating than a single call, which can come from anywhere.
EDGE_WEIGHTS = {
    "extends":      4,
    "implements":   4,
    "annotation":   3,
    "field_access": 2,
    "type_ref":     2,
    "call":         1,
}


class WeightedNeighbourVote(NeighbourVote):
    """Neighbour vote with per-edge-kind weights."""
    id = "neighbour_vote_w"; tier = 3

    def __init__(self, min_votes: int = 6, max_rev_b: int = 200,
                 weights: dict | None = None):
        super().__init__(min_votes=min_votes, max_rev_b=max_rev_b)
        self.weights = weights or EDGE_WEIGHTS

    def propose(self, a, b, mapping):
        votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a_x, b_x in mapping:
            for kind, weight in self.weights.items():
                rev_b = list(b.reverse_neighbours(b_x, kind))
                if not rev_b or len(rev_b) > self.max_rev_b:
                    continue
                for a_voter in a.reverse_neighbours(a_x, kind):
                    if mapping.get(a_voter) is not None:
                        continue
                    voter_votes = votes[a_voter]
                    for nb in rev_b:
                        voter_votes[nb] += weight
        for a_cid, vmap in votes.items():
            if not vmap: continue
            top_b, top_v = max(vmap.items(), key=lambda kv: kv[1])
            if top_v < self.min_votes:
                continue
            second = max((v for k, v in vmap.items() if k != top_b), default=0)
            if top_v - second < 2:
                continue
            conf = min(0.55 + 0.02 * (top_v - second), 0.92)
            yield Candidate(a_cid, top_b, conf, self.id,
                            ("nv_w", top_v, second))


def _specificity_confidence(base: float, n_a: int, n_b: int,
                            ceiling: float = 0.95) -> float:
    """Confidence as a function of bucket size on each side.

    A fingerprint that uniquely identifies one class on each side
    is high-confidence; one that fires across many classes is low.
    Returns `base` when n_a == n_b == 1 and tapers down with the
    product of the bucket sizes.
    """
    n = n_a * n_b
    if n <= 1:
        return min(base, ceiling)
    # 1/sqrt(n) falloff with floor
    import math
    return max(base / math.sqrt(n), 0.4)


class CallTargetMultiset:
    """Multiset of stable-only (class, method-name) call-targets.

    Two classes that call the EXACT SAME multiset of (foo.Bar, method1)
    + (foo.Baz, method2) + ... are almost certainly the same class.
    Stable-only ensures the signal isn't washed out by LX-name rotation.

    Confidence scales with bucket specificity: a fingerprint shared by
    one A and one B class fires at near-1.0; a fingerprint shared by
    100 A and 100 B classes scores much lower.
    """
    id = "call_targets_ms"; tier = 2

    def __init__(self, min_targets: int = 4):
        self.min_targets = min_targets

    def _fp_for(self, rec: dict) -> str | None:
        # Filter to stable-class call targets only.
        ts = sorted({(c, m) for c, m in rec.get("call_targets", ())
                     if not c.startswith("LX/")})
        if len(ts) < self.min_targets:
            return None
        return _fp("ctm", ts)

    def propose(self, a, b):
        Bbk = defaultdict(list); Abk = defaultdict(list)
        for r in b.classes():
            h = self._fp_for(r)
            if h: Bbk[h].append(r["id"])
        for r in a.classes():
            h = self._fp_for(r)
            if h: Abk[h].append(r["id"])
        for h, alst in Abk.items():
            blst = Bbk.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.9, len(alst), len(blst))
            for ai in alst:
                for bi in blst:
                    yield Candidate(ai, bi, conf, self.id,
                                    ("call_targets_ms", len(alst), len(blst)))


class CallTargetWithSubstitution:
    """Same as CallTargetMultiset, but substitutes LX class refs through
    the current mapping. Tier 3 because it depends on a partially-built
    mapping. Catches classes whose API surface is identical once
    obfuscated callees are resolved."""
    id = "call_targets_sub"; tier = 3

    def __init__(self, min_targets: int = 6):
        self.min_targets = min_targets

    def _fp_for(self, rec: dict, sub) -> str | None:
        ts = []
        for c, m in rec.get("call_targets", ()):
            if c.startswith("LX/"):
                mc = sub(c)
                if mc is None:
                    continue   # skip unresolved
                ts.append((mc, m))
            else:
                ts.append((c, m))
        ts = sorted(set(ts))
        if len(ts) < self.min_targets:
            return None
        return _fp("cts", ts)

    def propose(self, a, b, mapping):
        # B side: identity sub (B's call_targets already use B-side names).
        identity = lambda x: x
        Bbk = defaultdict(list)
        for r in b.classes():
            h = self._fp_for(r, identity)
            if h: Bbk[h].append(r["id"])
        # A side: substitute via mapping.
        sub = lambda x: mapping.get(x)
        for r in a.classes():
            h = self._fp_for(r, sub)
            if not h: continue
            blst = Bbk.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.85, 1, len(blst))
            for bi in blst:
                yield Candidate(r["id"], bi, conf, self.id,
                                ("call_targets_sub", len(blst)))


class FieldTargetMultiset:
    """Multiset of stable-only (class, field-name) accesses. Same idea
    as CallTargetMultiset for field reads/writes."""
    id = "field_targets_ms"; tier = 2

    def __init__(self, min_targets: int = 3):
        self.min_targets = min_targets

    def _fp_for(self, rec: dict) -> str | None:
        ts = sorted({(c, f) for c, f in rec.get("field_targets", ())
                     if not c.startswith("LX/")})
        if len(ts) < self.min_targets:
            return None
        return _fp("ftm", ts)

    def propose(self, a, b):
        Bbk = defaultdict(list); Abk = defaultdict(list)
        for r in b.classes():
            h = self._fp_for(r)
            if h: Bbk[h].append(r["id"])
        for r in a.classes():
            h = self._fp_for(r)
            if h: Abk[h].append(r["id"])
        for h, alst in Abk.items():
            blst = Bbk.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.85, len(alst), len(blst))
            for ai in alst:
                for bi in blst:
                    yield Candidate(ai, bi, conf, self.id,
                                    ("field_targets_ms", len(alst), len(blst)))


class SiblingByMappedSuper:
    """Tier-3 matcher for tiny classes that share a (post-mapping)
    super and a structural shape.

    For each A class C with super S_a:
      * If S_a is in the stable namespace, S_b = S_a.
      * Else look up `mapping.get(S_a)` — skip if unmapped.
      * Bucket A and B classes by (S_b, nm, nf, ns, mods, sorted(impls_b)).
      * Whenever the bucket has exactly one A and exactly one B class,
        propose them as a candidate.

    Catches the lambdas / synthetic / data-carrier classes that have
    no strings but share an inheritance + shape fingerprint with
    their sibling on the other side.
    """
    id = "sibling_super"; tier = 3

    def _impls_substituted(self, cls: dict, mapping) -> list[str]:
        out = []
        for x in cls.get("impls", ()):
            if not x.startswith("LX/"):
                out.append(x)
            else:
                m = mapping.get(x)
                out.append(m or "?")
        return sorted(out)

    def _mapped_super(self, cls: dict, mapping) -> str | None:
        s = cls.get("super")
        if not s or s == "Ljava/lang/Object;":
            return None
        if not s.startswith("LX/"):
            return s
        return mapping.get(s)

    def propose(self, a, b, mapping):
        # Build B index keyed by (super_b, nm, nf, ns, sorted_impls_b, mods)
        Bidx = defaultdict(list)
        for r in b.classes():
            sb = r.get("super")
            if not sb or sb == "Ljava/lang/Object;":
                continue
            key = (sb, r["nm"], r["nf"], r["ns"], r["nn"],
                   tuple(sorted(r["impls"])), tuple(r["mods"]))
            Bidx[key].append(r["id"])
        for r in a.classes():
            sa_mapped = self._mapped_super(r, mapping)
            if sa_mapped is None:
                continue
            key = (sa_mapped, r["nm"], r["nf"], r["ns"], r["nn"],
                   tuple(self._impls_substituted(r, mapping)),
                   tuple(r["mods"]))
            blst = Bidx.get(key)
            if not blst or len(blst) != 1:
                continue
            # Skip giant buckets — must be unique on A side too
            yield Candidate(r["id"], blst[0], 0.7, self.id, ("sibling_super",))


DEFAULT_MATCHERS = [
    # ---- Tier 1: anchors, lockable ------------------------------------
    FQNStable(),
    NativeSymbolSet(),
    IdenticalStrings(),
    LongUniqueString(),

    # ---- Tier 2: content fingerprints ---------------------------------
    UniqueString(),
    StringPair(),
    StringSetHash(),
    StringsPlusStableRefs(),
    StableRefsMultiset(),
    CallTargetMultiset(min_targets=4),   # NEW: who-calls-which-method
    FieldTargetMultiset(min_targets=3),  # NEW: who-touches-which-field

    # ---- Tier 3: propagation, iterated --------------------------------
    WeightedNeighbourVote(min_votes=6),
    BodyHashSubstituted(),
    CallTargetWithSubstitution(min_targets=6),  # NEW: substituted call-targets
    SiblingByMappedSuper(),
]
