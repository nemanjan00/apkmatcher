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
    tier = 3

    def __init__(self, min_votes: int = 6, max_rev_b: int = 200,
                 weights: dict | None = None):
        super().__init__(min_votes=min_votes, max_rev_b=max_rev_b)
        self.weights = weights or EDGE_WEIGHTS
        self.id = f"neighbour_vote_w_n{min_votes}"

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


class EnumValueNames:
    """For `enum` classes, match by the multiset of enum value names.

    Enums in Java keep their `name()` strings (used by reflection,
    JSON serialization, equals checks). Even when the class name
    rotates, an enum's value names like ['MAIN', 'CRITICAL_REPORT']
    are preserved verbatim across builds. The full multiset of
    value names is essentially a unique class fingerprint.

    Heuristic: a string in an enum's `strings` is likely a value
    name iff it's identifier-shaped (no spaces, no '/', no '.',
    no '%') and at least 2 chars. We don't need to identify
    value-name strings perfectly — even with some log/format
    strings mixed in, the SET equality test is still very
    discriminating because the rest of B's enums are different.
    """
    id = "enum_value_names"; tier = 2

    @staticmethod
    def _is_namelike(s: str) -> bool:
        if not s or len(s) < 2:
            return False
        if any(c in s for c in " /\\.%\"\n\t,:?"):
            return False
        return True

    def _fp_for(self, rec: dict) -> str | None:
        if "enum" not in rec.get("mods", ()):
            return None
        names = sorted({s for s in rec["strings"] if self._is_namelike(s)})
        if len(names) < 2:
            return None
        return _fp("env", names)

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
            if not blst:
                continue
            conf = _specificity_confidence(0.95, len(alst), len(blst), 0.97)
            for ai in alst:
                for bi in blst:
                    yield Candidate(ai, bi, conf, self.id,
                                    ("enum_values", len(alst), len(blst)))


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


class MethodCallSetSubstituted:
    """Tier 3. For each A class, compute a per-method 'API surface'
    fingerprint:

       method_fp = hash( substituted (target_class, member) calls,
                         substituted (target_class, field) accesses,
                         param/return types substituted,
                         branch count, string count )

    Class fingerprint = sorted multiset of per-method fingerprints.
    Match against the same B-side multiset (B's methods need no
    substitution — they're in B's own LX namespace).

    The crucial bit: substitution lifts A's bodies into a
    B-comparable namespace, so two classes that differ only by which
    LX names rotated still produce the same fingerprint once those
    LX names are mapped. Adds independent signal beyond
    `call_targets_sub` (which works at class-level multiset only).
    """
    id = "method_callset_sub"; tier = 3

    def __init__(self, min_methods: int = 2):
        self.min_methods = min_methods

    def _sub_ref(self, ref: str, mapping) -> str:
        if ref.startswith("LX/"):
            return mapping.get(ref) or ref
        return ref

    def _sub_sig(self, sig: str, mapping) -> str:
        out = []; i = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1: out.append(sig[i:]); break
                ref = sig[i:e+1]
                out.append(self._sub_ref(ref, mapping))
                i = e + 1; continue
            out.append(c); i += 1
        return "".join(out)

    def _method_fp(self, m: dict, mapping) -> str:
        cs = sorted({(self._sub_ref(c, mapping), n) for c, n in m["calls"]})
        fs = sorted({(self._sub_ref(c, mapping), n) for c, n in m["facc"]})
        sig_sub = self._sub_sig(m["sig"], mapping)
        return _fp("mfp", sig_sub, cs, fs, m["br"], len(m["strs"]))

    def _class_fp(self, rec: dict, mapping) -> str | None:
        ms = rec.get("methods", ())
        if len(ms) < self.min_methods:
            return None
        fps = sorted(self._method_fp(m, mapping) for m in ms)
        return _fp("cls", fps)

    def propose(self, a, b, mapping):
        # Identity sub for B side.
        class _IdMapping:
            def get(self, x): return None
        idmap = _IdMapping()
        Bidx = defaultdict(list)
        for r in b.classes():
            h = self._class_fp(r, idmap)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = self._class_fp(r, mapping)
            if not h: continue
            blst = Bidx.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.9, 1, len(blst))
            for bid in blst:
                yield Candidate(r["id"], bid, conf, self.id,
                                ("method_callset_sub", len(blst)))


class LineRefMultiset:
    """Tier 2. Per the source-line invariant: when a class invokes a
    stable framework class at a specific line in the source file,
    both builds of that class do it at (roughly) the same line.

    Fingerprint = multiset of (source_line, stable_target_class)
    tuples. Restricted to stable targets because LX names rotate.

    High precision when fingerprint size >= `min_refs`. Specificity
    scaling deals with classes that share boilerplate line-refs.
    """
    id = "line_refs_ms"; tier = 2

    def __init__(self, min_refs: int = 4):
        self.min_refs = min_refs

    def _fp(self, rec: dict) -> str | None:
        refs = sorted(set(map(tuple, rec.get("line_refs", ()))))
        if len(refs) < self.min_refs:
            return None
        return _fp("lrm", refs)

    def propose(self, a, b):
        Bidx = defaultdict(list); Aidx = defaultdict(list)
        for r in b.classes():
            h = self._fp(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            h = self._fp(r)
            if h: Aidx[h].append(r["id"])
        for h, alst in Aidx.items():
            blst = Bidx.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.92, len(alst), len(blst))
            for ai in alst:
                for bi in blst:
                    yield Candidate(ai, bi, conf, self.id,
                                    ("line_refs", len(alst), len(blst)))


class JaccardStrings:
    """Tier 2. Fuzzy match by Jaccard similarity over the string set.

    Catches classes whose string content shifted slightly between
    builds — a few new log messages added or one renamed — but the
    bulk of the strings overlap. Inverted-index over individual
    strings to find candidate pairs instead of brute force.

    Conservative: requires Jaccard >= `min_jaccard` AND at least
    `min_overlap` shared distinct strings. Confidence scales with
    Jaccard.
    """
    tier = 2

    def __init__(self, min_jaccard: float = 0.7, min_overlap: int = 3):
        self.min_jaccard = min_jaccard
        self.min_overlap = min_overlap
        self.id = f"jaccard_strings_j{int(min_jaccard*10)}_o{min_overlap}"

    def propose(self, a, b):
        # Candidate-set generation: for each A class with >= min_overlap
        # strings, find B classes that share at least min_overlap of
        # them. Use the inverted string index on B.
        for ra in a.classes():
            sa = {s for s in ra["strings"] if len(s) >= 4}
            if len(sa) < self.min_overlap:
                continue
            counts: dict[str, int] = defaultdict(int)
            for s in sa:
                # Use B's classes_containing_string
                for bid in b.classes_containing_string(s):
                    counts[bid] += 1
            # Keep only candidates with >= min_overlap shared
            for bid, n in counts.items():
                if n < self.min_overlap:
                    continue
                rb = b.get(bid)
                if rb is None: continue
                sb = {s for s in rb["strings"] if len(s) >= 4}
                if not sb:
                    continue
                jac = n / len(sa | sb)
                if jac < self.min_jaccard:
                    continue
                # Confidence: jaccard mapped from [min, 1] to [0.6, 0.92]
                conf = 0.6 + 0.32 * (jac - self.min_jaccard) / (1 - self.min_jaccard)
                yield Candidate(ra["id"], bid, conf, self.id,
                                ("jaccard_strings", round(jac, 2), n))


class ExtendedByLockStep:
    """Tier-3, pin a super class by the (mapped) classes that extend it.

    For each unmatched A class C: look at A.reverse_neighbours(C, "extends")
    — every class that DECLARES C as its super. Among those that are
    already mapped, look at the B-side image's super. If all the
    B-side supers point to the same (unmatched) B class, that's C's
    match.

    Cheaper and more conservative than the general ReverseLockStep
    because it filters to a single edge kind. Targets the
    'unmatched super' deadlock case where many extends-children share
    a parent that has no other distinguishing signal of its own.
    """
    tier = 3

    def __init__(self, min_mapped: int = 2):
        self.min_mapped = min_mapped
        self.id = f"extended_by_lockstep_n{min_mapped}"

    def propose(self, a, b, mapping):
        for cid in a.ids():
            if mapping.get(cid) is not None:
                continue
            children = list(a.reverse_neighbours(cid, "extends"))
            mapped_children = [mapping.get(ch) for ch in children
                               if mapping.get(ch) is not None]
            if len(mapped_children) < self.min_mapped:
                continue
            # Collect each mapped child's B-side super.
            b_supers = set()
            for mc in mapped_children:
                rb = b.get(mc)
                if rb is None: continue
                bs = rb.get("super")
                if bs:
                    b_supers.add(bs)
                    if len(b_supers) > 1: break
            if len(b_supers) != 1:
                continue
            (bsuper,) = b_supers
            # B-side super must itself be unmatched (we're proposing it as C's match).
            if mapping.inverse(bsuper) is not None:
                continue
            conf = min(0.85 + 0.03 * len(mapped_children), 0.97)
            yield Candidate(cid, bsuper, conf, self.id,
                            ("extended_by", len(mapped_children)))


class MappedNeighbourFingerprint:
    """Tier-3. For each unmatched A class, build a sorted multiset of
    its outgoing neighbours after mapping substitution. For B side,
    use the raw outgoing neighbours. If the multisets match exactly
    (and the class shape agrees), propose.

    Catches the case 'all my refs are mapped, just find the B class
    with matching mapped-target set'. Targets the ~4500 unmatched
    classes whose every reference resolves through the mapping but
    whose own identity isn't pinnable by other tier-3 matchers.

    Includes (super, sorted_impls, mods, nm, nf) shape constraint
    in the fingerprint to avoid over-matching tiny classes.
    """
    id = "mapped_nb_fp"; tier = 3

    def __init__(self, min_neighbours: int = 3, min_mapped_ratio: float = 0.7):
        self.min_neighbours = min_neighbours
        self.min_mapped_ratio = min_mapped_ratio

    def _sub_super(self, s, mapping):
        if not s or s == "Ljava/lang/Object;": return s
        if s.startswith("LX/"): return mapping.get(s) or s
        return s

    def _sub_impls(self, impls, mapping):
        out = []
        for x in impls:
            if x.startswith("LX/"):
                m = mapping.get(x)
                if m is None: return None
                out.append(m)
            else:
                out.append(x)
        return tuple(sorted(out))

    def _fp_a(self, ra: dict, mapping):
        nbs = sorted(set(ra["calls"] + ra["facc"] + ra["trefs"]))
        if len(nbs) < self.min_neighbours: return None
        # Substitute LX refs through mapping. Skip if too many unmapped.
        sub = []; mapped = 0
        for n in nbs:
            if n.startswith("LX/"):
                m = mapping.get(n)
                if m is None: continue
                sub.append(m); mapped += 1
            else:
                sub.append(n); mapped += 1
        if mapped / len(nbs) < self.min_mapped_ratio:
            return None
        sub_sorted = tuple(sorted(set(sub)))
        if len(sub_sorted) < self.min_neighbours:
            return None
        impls = self._sub_impls(ra.get("impls", ()), mapping)
        if impls is None: impls = ()
        return _fp("mnf",
                   sub_sorted,
                   self._sub_super(ra.get("super"), mapping),
                   impls,
                   ra["nm"], ra["nf"], ra["nn"],
                   tuple(ra["mods"]))

    def _fp_b(self, rb: dict):
        nbs = sorted(set(rb["calls"] + rb["facc"] + rb["trefs"]))
        if len(nbs) < self.min_neighbours: return None
        return _fp("mnf",
                   tuple(nbs),
                   rb.get("super") or "",
                   tuple(sorted(rb.get("impls", ()))),
                   rb["nm"], rb["nf"], rb["nn"],
                   tuple(rb["mods"]))

    def propose(self, a, b, mapping):
        Bidx = defaultdict(list)
        for r in b.classes():
            if not r["id"].startswith("LX/"):
                continue
            if mapping.inverse(r["id"]) is not None:
                continue
            h = self._fp_b(r)
            if h: Bidx[h].append(r["id"])
        for r in a.classes():
            if not r["id"].startswith("LX/"):
                continue
            if mapping.get(r["id"]) is not None:
                continue
            h = self._fp_a(r, mapping)
            if not h: continue
            blst = Bidx.get(h)
            if not blst: continue
            conf = _specificity_confidence(0.85, 1, len(blst))
            for bid in blst:
                yield Candidate(r["id"], bid, conf, self.id,
                                ("mapped_nb_fp", len(blst)))


class MethodWalk:
    """Tier-3 method-level graph walk.

    For each confirmed class pair (A, A'), pair their methods by
    substituted-signature equality, falling back to a greedy
    best-match-by-call-set similarity. For each paired method
    (M_a, M_a'), walk their per-method call lists in parallel:

      For i in [0..min(len(calls_a), len(calls_a'))):
          (cls_a, name_a) = M_a.calls[i]
          (cls_b, name_b) = M_a'.calls[i]
          if cls_a is unmatched LX, cls_b is unmatched LX, and the
          method names are identical (or both obfuscated single
          char), record an inference cls_a -> cls_b.

    Tally inferences across all method-pairs in all confirmed
    class-pairs. Propose pairs whose inference count is high and
    whose runner-up is at least N votes lower.

    This is the 'lambdas referenced in only one place' case the
    user pointed out: a lambda L_a is only called from a single
    method M of class C. If C maps to C' and M maps to M' (by
    signature equality), and M' calls L_b at the same body
    position, then L_a -> L_b is forced.
    """
    id = "method_walk"; tier = 3

    def __init__(self, min_inferences: int = 2, min_margin: int = 1):
        self.min_inferences = min_inferences
        self.min_margin = min_margin

    def _sub_sig(self, sig: str, mapping) -> str:
        out = []; i = 0
        while i < len(sig):
            c = sig[i]
            if c == "L":
                e = sig.find(";", i)
                if e == -1: out.append(sig[i:]); break
                ref = sig[i:e+1]
                if ref.startswith("LX/"):
                    out.append(mapping.get(ref) or ref)
                else:
                    out.append(ref)
                i = e + 1; continue
            out.append(c); i += 1
        return "".join(out)

    def _pair_methods(self, ma_list, mb_list, mapping):
        """Greedy method pairing within a confirmed class pair.
        Pass 1: substituted-signature equality. Pass 2: best
        remaining matches by (calls, facc) Jaccard. Returns list
        of (ma, mb) pairs."""
        out = []
        b_by_sub_sig = defaultdict(list)
        for mb in mb_list:
            b_by_sub_sig[mb["sig"]].append(mb)
        used_b = set()
        leftover_a = []
        for ma in ma_list:
            sub = self._sub_sig(ma["sig"], mapping)
            cands = b_by_sub_sig.get(sub, ())
            picked = None
            for mb in cands:
                mid = id(mb)
                if mid in used_b: continue
                picked = mb; used_b.add(mid); break
            if picked is not None:
                out.append((ma, picked))
            else:
                leftover_a.append(ma)
        # Greedy pass on leftovers
        leftover_b = [mb for mb in mb_list if id(mb) not in used_b]
        for ma in leftover_a:
            ma_calls = set(map(tuple, ma["calls"]))
            ma_facc = set(map(tuple, ma["facc"]))
            best = None; best_s = 0.0
            for mb in leftover_b:
                if id(mb) in used_b: continue
                mb_calls = set(map(tuple, mb["calls"]))
                mb_facc = set(map(tuple, mb["facc"]))
                u = (ma_calls | mb_calls); ix = (ma_calls & mb_calls)
                jc = len(ix) / max(1, len(u))
                u2 = (ma_facc | mb_facc); ix2 = (ma_facc & mb_facc)
                jf = len(ix2) / max(1, len(u2))
                s = 0.6 * jc + 0.4 * jf
                if s > best_s:
                    best_s = s; best = mb
            if best is not None and best_s >= 0.5:
                out.append((ma, best))
                used_b.add(id(best))
        return out

    def propose(self, a, b, mapping):
        votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for a_cid, b_cid in mapping:
            ra = a.get(a_cid); rb = b.get(b_cid)
            if not ra or not rb: continue
            ma_list = ra.get("methods", ())
            mb_list = rb.get("methods", ())
            if not ma_list or not mb_list: continue
            pairs = self._pair_methods(ma_list, mb_list, mapping)
            for ma, mb in pairs:
                # Walk parallel call lists
                ma_calls = ma.get("calls", ())
                mb_calls = mb.get("calls", ())
                # Pair by method-name equality + position proximity:
                # multiset of LX callees per method.
                a_lx = [(cls, nm) for cls, nm in ma_calls if cls.startswith("LX/")
                        and mapping.get(cls) is None]
                b_lx = [(cls, nm) for cls, nm in mb_calls if cls.startswith("LX/")
                        and mapping.inverse(cls) is None]
                # If both are size 1, infer directly
                if len(a_lx) == 1 and len(b_lx) == 1:
                    if a_lx[0][1] == b_lx[0][1]:
                        votes[a_lx[0][0]][b_lx[0][0]] += 2
                    else:
                        votes[a_lx[0][0]][b_lx[0][0]] += 1
                # General: pair by method-name match
                else:
                    a_by_name = defaultdict(list)
                    b_by_name = defaultdict(list)
                    for cls, nm in a_lx: a_by_name[nm].append(cls)
                    for cls, nm in b_lx: b_by_name[nm].append(cls)
                    for nm, alist in a_by_name.items():
                        blist = b_by_name.get(nm)
                        if not blist: continue
                        if len(alist) == 1 and len(blist) == 1:
                            votes[alist[0]][blist[0]] += 1
        for a_cid, vmap in votes.items():
            if not vmap: continue
            top_b, top_v = max(vmap.items(), key=lambda kv: kv[1])
            if top_v < self.min_inferences: continue
            second = max((v for k, v in vmap.items() if k != top_b), default=0)
            if top_v - second < self.min_margin: continue
            conf = min(0.65 + 0.04 * (top_v - second), 0.92)
            yield Candidate(a_cid, top_b, conf, self.id,
                            ("method_walk", top_v, second))


class DisambiguatingLockStep:
    """Tier-3, fallback when LockStep's intersection has 2-N candidates.

    Plain LockStep requires the intersection of B-side reverse-neighbour
    sets to have exactly one unmatched element. When the intersection is
    small (2..max_set), we can ask: which of those candidates has the
    best content match against the A class?

    Lower confidence than strict LockStep because we're picking a
    winner instead of finding a forced match. Conservative content
    threshold (>= 0.6) and a strict margin requirement (best > second
    + 0.1) means at-most-one weak candidate ever fires.
    """
    tier = 3

    def __init__(self, min_mapped: int = 3, max_set: int = 5,
                 min_score: float = 0.6, min_margin: float = 0.1):
        self.min_mapped = min_mapped
        self.max_set = max_set
        self.min_score = min_score
        self.min_margin = min_margin
        self.id = f"disambig_lockstep_n{min_mapped}"

    def propose(self, a, b, mapping):
        rev_cache: dict[str, set] = {}
        def rev(bt: str) -> set:
            r = rev_cache.get(bt)
            if r is None:
                r = set(b.reverse_neighbours(bt))
                rev_cache[bt] = r
            return r

        for cid in a.ids():
            if mapping.get(cid) is not None: continue
            ra = a.get(cid)
            seen = set(); mapped_targets = []
            for nb in a.neighbours(cid):
                if nb in seen: continue
                seen.add(nb)
                mb = mapping.get(nb)
                if mb is not None:
                    mapped_targets.append(mb)
            if len(mapped_targets) < self.min_mapped:
                continue

            seed_set = None; seed_target = None
            for bt in mapped_targets:
                rs = rev(bt)
                if not rs or len(rs) > 500: continue
                if seed_set is None or len(rs) < len(seed_set):
                    seed_set = rs; seed_target = bt
            if seed_set is None: continue
            cands = set(seed_set)
            for bt in mapped_targets:
                if bt == seed_target: continue
                cands &= rev(bt)
                if not cands: break
            cands = {x for x in cands if mapping.inverse(x) is None}
            if len(cands) < 2 or len(cands) > self.max_set:
                continue

            scored = []
            for bcid in cands:
                rb = b.get(bcid)
                if rb is None: continue
                s = _content_score(ra, rb, mapping)
                scored.append((s, bcid))
            scored.sort(reverse=True)
            if len(scored) < 2: continue
            top_s, top_b = scored[0]
            second_s = scored[1][0]
            if top_s < self.min_score: continue
            if top_s - second_s < self.min_margin: continue
            yield Candidate(cid, top_b, 0.65 + (top_s - second_s) * 0.5,
                            self.id, ("disambig", round(top_s, 2)))


class ImplementedByLockStep:
    """Tier-3, pin an interface by the (mapped) classes that implement it.
    Mirror of ExtendedByLockStep for interface implementations.
    """
    tier = 3

    def __init__(self, min_mapped: int = 3):
        self.min_mapped = min_mapped
        self.id = f"implemented_by_lockstep_n{min_mapped}"

    def propose(self, a, b, mapping):
        for cid in a.ids():
            if mapping.get(cid) is not None:
                continue
            implementors = list(a.reverse_neighbours(cid, "implements"))
            mapped = [mapping.get(c) for c in implementors
                      if mapping.get(c) is not None]
            if len(mapped) < self.min_mapped:
                continue
            # Each mapped B implementor declares some impls. The
            # candidate B interface(s) are the ones present in EVERY
            # mapped implementor's impls list. (Intersection.)
            cand: set | None = None
            for mc in mapped:
                rb = b.get(mc)
                if rb is None: continue
                bset = set(rb.get("impls", ()))
                if not bset:
                    cand = set(); break
                cand = bset if cand is None else (cand & bset)
                if not cand:
                    break
            if not cand:
                continue
            # Restrict to unmatched B classes.
            cand = {x for x in cand if mapping.inverse(x) is None}
            if len(cand) != 1:
                continue
            (bcid,) = cand
            conf = min(0.85 + 0.02 * len(mapped), 0.97)
            yield Candidate(cid, bcid, conf, self.id,
                            ("implemented_by", len(mapped)))


class ReverseLockStep:
    """Tier-3, mirrors LockStep but uses INCOMING edges.

    For each unmatched A class C, look at A.reverse_neighbours(C) —
    the classes that reference C. Among those that are mapped, get
    their B-side images. Each such image references some B classes;
    the *intersection* of those reference sets is the set of B
    classes referenced by every B-side analogue of C's references.
    If exactly one unmatched B class is in that intersection, it's
    C's match.

    This is the dual of LockStep — useful for marker interfaces,
    listener types, abstract bases that have many implementors but
    few outgoing edges of their own.
    """
    tier = 3

    def __init__(self, min_mapped: int = 4, max_per_source: int = 500):
        self.min_mapped = min_mapped
        self.max_per_source = max_per_source
        self.id = f"reverse_lockstep_n{min_mapped}"

    def propose(self, a, b, mapping):
        fwd_cache: dict[str, set] = {}
        def fwd(bs: str) -> set:
            r = fwd_cache.get(bs)
            if r is None:
                r = set(b.neighbours(bs))
                fwd_cache[bs] = r
            return r

        for cid in a.ids():
            if mapping.get(cid) is not None:
                continue
            seen = set()
            mapped_b_sources: list[str] = []
            for nb in a.reverse_neighbours(cid):
                if nb in seen:
                    continue
                seen.add(nb)
                mb = mapping.get(nb)
                if mb is not None:
                    mapped_b_sources.append(mb)
            if len(mapped_b_sources) < self.min_mapped:
                continue

            seed_source = None; seed_set = None
            for bs in mapped_b_sources:
                fs = fwd(bs)
                if not fs or len(fs) > self.max_per_source:
                    continue
                if seed_set is None or len(fs) < len(seed_set):
                    seed_set = fs; seed_source = bs
            if seed_set is None:
                continue
            cands = set(seed_set)
            for bs in mapped_b_sources:
                if bs == seed_source:
                    continue
                fs = fwd(bs)
                if not fs:
                    continue
                cands &= fs
                if not cands:
                    break
            cands = {x for x in cands if mapping.inverse(x) is None}
            if len(cands) != 1:
                continue
            (bcid,) = cands
            conf = min(0.85 + 0.02 * len(mapped_b_sources), 0.97)
            yield Candidate(cid, bcid, conf, self.id,
                            ("reverse_lockstep", len(mapped_b_sources)))


def _content_score(ra: dict, rb: dict, mapping) -> float:
    """Cheap pairwise compatibility: shape + string overlap + stable-ref
    overlap + super agreement (after substitution). Range [0, 1]."""
    score = 0.0
    weight = 0.0

    # Shape (modifier flags + counts)
    sa = set(ra["mods"]); sb = set(rb["mods"])
    for k in ("interface", "enum", "annotation", "abstract", "final"):
        if (k in sa) == (k in sb):
            score += 0.5
        weight += 0.5
    if ra["nm"] == rb["nm"]: score += 1.0
    weight += 1.0
    if ra["nf"] == rb["nf"]: score += 1.0
    weight += 1.0

    # String Jaccard
    ssa = {s for s in ra["strings"] if len(s) >= 4}
    ssb = {s for s in rb["strings"] if len(s) >= 4}
    if ssa or ssb:
        jac = len(ssa & ssb) / max(1, len(ssa | ssb))
        score += 3.0 * jac
        weight += 3.0

    # Stable refs Jaccard
    sta = {r for r in ra["calls"] + ra["facc"] + ra["trefs"]
           if not r.startswith("LX/")}
    stb = {r for r in rb["calls"] + rb["facc"] + rb["trefs"]
           if not r.startswith("LX/")}
    if sta or stb:
        jac = len(sta & stb) / max(1, len(sta | stb))
        score += 2.0 * jac
        weight += 2.0

    # Super agreement (after sub)
    super_a = ra.get("super") or ""
    super_b = rb.get("super") or ""
    if super_a.startswith("LX/"):
        super_a = mapping.get(super_a) or super_a
    if super_a == super_b:
        score += 1.5
    weight += 1.5

    return score / weight if weight else 0.0


_LOCKSTEP_ID_FMT = "lockstep_n{}"


class LockStep:
    """Tier-3, very high precision.

    For each unmatched A class C with `min_mapped` or more
    *unique* mapped forward neighbours:
      * Compute the multiset of B-side images of those neighbours.
      * For each B-side image, gather B's reverse-neighbours.
      * The intersection across all those reverse-neighbour sets
        is the set of B classes that reference EVERY one of C's
        mapped neighbours' images.
      * If the intersection contains exactly one unmatched B class,
        that class is C's match (lock-step constraint satisfied:
        all but possibly one outgoing edge agrees).

    This is the spec's "lock-step" propagation: when most of a
    class's neighbours are pinned, the class itself is pinned.
    Conservative — emits at most one candidate per A class, only
    when the intersection is unambiguous.
    """
    tier = 3

    def __init__(self, min_mapped: int = 4, max_per_target: int = 500):
        self.min_mapped = min_mapped
        self.max_per_target = max_per_target
        self.id = _LOCKSTEP_ID_FMT.format(min_mapped)

    def propose(self, a, b, mapping):
        # Cache reverse-neighbour sets — the same B target appears in
        # many A classes' neighbour lists; recomputing is the cost.
        rev_cache: dict[str, set] = {}
        def rev(bt: str) -> set:
            r = rev_cache.get(bt)
            if r is None:
                r = set(b.reverse_neighbours(bt))
                rev_cache[bt] = r
            return r

        for cid in a.ids():
            if mapping.get(cid) is not None:
                continue
            seen = set()
            mapped_b_targets: list[str] = []
            for nb in a.neighbours(cid):
                if nb in seen:
                    continue
                seen.add(nb)
                mb = mapping.get(nb)
                if mb is not None:
                    mapped_b_targets.append(mb)
            if len(mapped_b_targets) < self.min_mapped:
                continue

            # Start with the smallest reverse-neighbour set as the seed.
            seed_target = None
            seed_set = None
            for bt in mapped_b_targets:
                rs = rev(bt)
                if not rs or len(rs) > self.max_per_target:
                    continue
                if seed_set is None or len(rs) < len(seed_set):
                    seed_set = rs
                    seed_target = bt
            if seed_set is None:
                continue
            cands = set(seed_set)
            for bt in mapped_b_targets:
                if bt == seed_target:
                    continue
                rs = rev(bt)
                if not rs:
                    continue
                cands &= rs
                if not cands:
                    break
            cands = {x for x in cands if mapping.inverse(x) is None}
            if len(cands) != 1:
                continue
            (bcid,) = cands
            conf = min(0.85 + 0.02 * len(mapped_b_targets), 0.97)
            yield Candidate(cid, bcid, conf, self.id,
                            ("lockstep", len(mapped_b_targets)))


class SiblingByMappedInterfaces:
    """Tier-3. Classes that implement the SAME mapped interface set
    AND share a structural shape are typically siblings in the
    same hierarchy. Catches listener/callback / data-transfer-object
    classes that have no super beyond Object but implement one or
    more (mapped) interfaces.

    Bucket key: (sorted_mapped_impls, nm, nf, ns, mods).
    Yields candidates when both A and B side bucket has exactly one
    class.
    """
    id = "sibling_impls"; tier = 3

    def _impls_substituted(self, cls: dict, mapping) -> tuple | None:
        out = []
        for x in cls.get("impls", ()):
            if not x.startswith("LX/"):
                out.append(x)
            else:
                m = mapping.get(x)
                if m is None:
                    return None  # don't fingerprint until everything resolves
                out.append(m)
        if not out:
            return None
        return tuple(sorted(out))

    def propose(self, a, b, mapping):
        Bidx = defaultdict(list)
        for r in b.classes():
            if not r["impls"]:
                continue
            key = (tuple(sorted(r["impls"])), r["nm"], r["nf"], r["ns"],
                   r["nn"], tuple(r["mods"]))
            Bidx[key].append(r["id"])
        for r in a.classes():
            ki = self._impls_substituted(r, mapping)
            if ki is None:
                continue
            key = (ki, r["nm"], r["nf"], r["ns"], r["nn"], tuple(r["mods"]))
            blst = Bidx.get(key)
            if not blst or len(blst) != 1:
                continue
            yield Candidate(r["id"], blst[0], 0.7, self.id, ("sibling_impls",))


class SiblingByMappedSuperLoose:
    """Tier-3, relaxed variant of SiblingByMappedSuper.

    Allows method/field counts to differ by ±1 between A and B
    (small refactors that add or remove a single helper method).
    Still requires (mapped_super, mods, sorted_impls_substituted).
    Yields candidates only when the relaxed bucket on each side
    contains a singleton. Lower confidence than the strict match.
    """
    id = "sibling_super_loose"; tier = 3

    def _impls_substituted(self, cls: dict, mapping) -> tuple | None:
        out = []
        for x in cls.get("impls", ()):
            if not x.startswith("LX/"):
                out.append(x)
            else:
                m = mapping.get(x)
                if m is None: return None
                out.append(m)
        return tuple(sorted(out))

    def _mapped_super(self, cls: dict, mapping):
        s = cls.get("super")
        if not s or s == "Ljava/lang/Object;":
            return None
        if not s.startswith("LX/"):
            return s
        return mapping.get(s)

    def propose(self, a, b, mapping):
        Bbk = defaultdict(list)
        for r in b.classes():
            sb = r.get("super")
            if not sb or sb == "Ljava/lang/Object;":
                continue
            key = (sb, tuple(sorted(r["impls"])), tuple(r["mods"]))
            Bbk[key].append(r)
        for r in a.classes():
            sa = self._mapped_super(r, mapping)
            if sa is None:
                continue
            ki = self._impls_substituted(r, mapping)
            if ki is None:
                continue
            key = (sa, ki, tuple(r["mods"]))
            blst = Bbk.get(key)
            if not blst:
                continue
            # Filter by shape ± 1
            ok = [b_rec for b_rec in blst
                  if abs(b_rec["nm"] - r["nm"]) <= 1
                  and abs(b_rec["nf"] - r["nf"]) <= 1
                  and b_rec["nn"] == r["nn"]
                  and mapping.inverse(b_rec["id"]) is None]
            if len(ok) != 1:
                continue
            yield Candidate(r["id"], ok[0]["id"], 0.62, self.id,
                            ("sibling_loose",))


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
    CallTargetMultiset(min_targets=4),
    FieldTargetMultiset(min_targets=3),
    EnumValueNames(),
    JaccardStrings(min_jaccard=0.7, min_overlap=3),
    LineRefMultiset(min_refs=4),

    # ---- Tier 3: propagation, iterated --------------------------------
    LockStep(min_mapped=4),
    LockStep(min_mapped=3),
    LockStep(min_mapped=2),
    LockStep(min_mapped=1),
    DisambiguatingLockStep(min_mapped=3),
    MappedNeighbourFingerprint(min_neighbours=3, min_mapped_ratio=0.7),
    MethodWalk(min_inferences=2, min_margin=1),
    ReverseLockStep(min_mapped=4),
    ReverseLockStep(min_mapped=3),
    ReverseLockStep(min_mapped=2),
    ReverseLockStep(min_mapped=1),
    ExtendedByLockStep(min_mapped=2),
    ExtendedByLockStep(min_mapped=1),
    ImplementedByLockStep(min_mapped=3),
    ImplementedByLockStep(min_mapped=2),
    ImplementedByLockStep(min_mapped=1),
    MethodCallSetSubstituted(),
    WeightedNeighbourVote(min_votes=4),
    WeightedNeighbourVote(min_votes=2),  # Aggressive pass after others run
    BodyHashSubstituted(),
    CallTargetWithSubstitution(min_targets=6),
    SiblingByMappedSuper(),
    SiblingByMappedSuperLoose(),
    # SiblingByMappedInterfaces() — tried but regressed quality
    # (matches lambdas that look alike; LockStep already covers
    # the cases it gets right).
]
