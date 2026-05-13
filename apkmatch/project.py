"""InMemoryProject — concrete ProjectView backed by dicts.

A project is one APK's worth of classes plus inverted indexes for the
queries matchers and validators make through the ProjectView interface.
Index construction is eager and one-shot at load time; everything after
that is dict/set lookups.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterator, Optional

# R8 lambda-merging optimisation collapses many independent
# Function0/Function1/Function2/... lambdas into a single class whose
# `invoke()` dispatches on an integer field. Two such "merged" classes
# from different builds rarely correspond to the same logical lambda —
# the merger groups them on internal heuristics that change build to
# build. Match these as classes at your peril.
_KOTLIN_FN_IMPL = re.compile(r"^Lkotlin/jvm/functions/Function\d+;$")


def _stable(cid: str) -> bool:
    """Returns True if this class name lives in a namespace that survives
    obfuscation rotation across builds.

    Instagram's R8 setup rotates only LX/... names. Everything else
    (framework, Kotlin, OSS libs, kept Instagram packages) is stable.
    Treat the stable namespaces as anchors when matching.
    """
    return not cid.startswith("LX/")


class InMemoryProject:
    """ProjectView implementation. See apkmatch.interfaces.ProjectView."""

    def __init__(self, records: list[dict]):
        self._classes: dict[str, dict] = {r["id"]: r for r in records}

        # Inverted indexes used by tier-1 / tier-2 matchers.
        self._by_string: dict[str, list[str]] = defaultdict(list)
        self._by_native_sym: dict[str, list[str]] = defaultdict(list)
        self._by_annotation: dict[str, list[str]] = defaultdict(list)
        self._by_super: dict[str, list[str]] = defaultdict(list)
        self._str_freq: dict[str, int] = defaultdict(int)

        # Forward / reverse edge indexes by kind. Edges are encoded as
        # plain (src, dst) tuples; kind is the dict key.
        self._fwd: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._rev: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        for r in records:
            cid = r["id"]
            seen = set()
            for s in r["strings"]:
                if s in seen or len(s) < 4:
                    continue
                seen.add(s)
                self._by_string[s].append(cid)
                self._str_freq[s] += 1
            for sym in r["native_syms"]:
                self._by_native_sym[sym].append(cid)
            for a in r["anns"]:
                self._by_annotation[a].append(cid)
            if r.get("super"):
                self._by_super[r["super"]].append(cid)

            def add(kind: str, target: str):
                if target in self._classes:  # internal edge only
                    self._fwd[kind][cid].add(target)
                    self._rev[kind][target].add(cid)

            if r.get("super"):  add("extends", r["super"])
            for x in r["impls"]: add("implements", x)
            for x in r["calls"]: add("call", x)
            for x in r["facc"]:  add("field_access", x)
            for x in r["trefs"]: add("type_ref", x)
            for x in r["anns"]:  add("annotation", x)

    # --- ProjectView ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._classes)

    def classes(self) -> Iterator[dict]:
        return iter(self._classes.values())

    def ids(self) -> Iterator[str]:
        return iter(self._classes.keys())

    def get(self, cid: str) -> Optional[dict]:
        return self._classes.get(cid)

    def neighbours(self, cid: str, kind: Optional[str] = None) -> Iterator[str]:
        if kind is not None:
            # Note: must `yield from`, NOT `return iter(...)` — this
            # function contains a yield in the no-kind branch, which
            # makes Python treat the entire function as a generator;
            # a `return value` inside a generator is silently ignored.
            yield from self._fwd[kind].get(cid, ())
            return
        seen = set()
        for k_edges in self._fwd.values():
            for t in k_edges.get(cid, ()):
                if t not in seen:
                    seen.add(t)
                    yield t

    def reverse_neighbours(self, cid: str, kind: Optional[str] = None) -> Iterator[str]:
        if kind is not None:
            yield from self._rev[kind].get(cid, ())
            return
        seen = set()
        for k_edges in self._rev.values():
            for s in k_edges.get(cid, ()):
                if s not in seen:
                    seen.add(s)
                    yield s

    def classes_containing_string(self, s: str) -> Iterator[str]:
        return iter(self._by_string.get(s, ()))

    def classes_with_native_symbol(self, sym: str) -> Iterator[str]:
        return iter(self._by_native_sym.get(sym, ()))

    def classes_with_annotation(self, ann: str) -> Iterator[str]:
        return iter(self._by_annotation.get(ann, ()))

    def string_frequency(self, s: str) -> int:
        return self._str_freq.get(s, 0)

    # --- helpers (concrete-only, not part of the protocol) --------------

    def stable_refs(self, cid: str) -> list[str]:
        """Type refs touched by this class, filtered to the stable
        (non-rotating) namespace. Foundational input for several
        matchers and validators."""
        r = self._classes.get(cid)
        if not r:
            return []
        out: list[str] = []
        for x in [r.get("super"), *r["impls"], *r["calls"],
                  *r["facc"], *r["trefs"], *r["anns"]]:
            if x and _stable(x):
                out.append(x)
        return out

    def is_obfuscated(self, cid: str) -> bool:
        return not _stable(cid)

    def is_lambda_merge(self, cid: str) -> bool:
        """True iff this class is an R8 lambda-merge container.

        Signature: implements `kotlin.jvm.functions.FunctionN`, has an
        `<init>(I...)V` ctor (the integer is the lambda discriminator)
        and an `invoke()` method whose body has `>= 5` branches (the
        dispatch switch). The 5-branch floor is conservative — a real
        per-callsite lambda almost never has more than a couple of
        branches.
        """
        r = self._classes.get(cid)
        if not r:
            return False
        if not any(_KOTLIN_FN_IMPL.match(i) for i in r.get("impls", ())):
            return False
        has_int_ctor = False
        invoke_branches = 0
        for m in r.get("methods", ()):
            sig = m.get("sig", "")
            if sig.startswith("<init>"):
                params = sig[sig.find("(") + 1: sig.find(")")]
                if "I" in params:
                    has_int_ctor = True
            if m.get("name") == "invoke":
                invoke_branches = max(invoke_branches, m.get("br", 0))
        return has_int_ctor and invoke_branches >= 5
