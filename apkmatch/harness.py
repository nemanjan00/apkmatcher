"""Test harness — `python -m apkmatch.harness`.

Evaluates a mapping produced by `apkmatch.cli match` without external
ground truth. The harness emits a JSON report of self-consistency
metrics, designed so a higher score implies a better algorithm.

Metrics:

  one_to_one_ratio
        Fraction of mappings where A→B is unique on both sides.
        Trivially 1.0 by construction of the engine, but recomputed
        here as a sanity check that the JSON wasn't tampered with.

  multi_matcher_agreement
        Of the confirmed pairs, the histogram of how many independent
        matchers fired on each pair. More matchers → more confidence.
        Score = mean # of matchers per pair.

  anchor_recovery
        For every A class in a stable (non-LX) namespace whose FQN also
        exists in B, did the algorithm map A → A? This is the only
        ground-truth signal available without external labels.
        Score = precision (correct stable-FQN matches / proposed
        stable-FQN matches) and recall (correct / total possible).

  neighbour_consistency
        For each confirmed (A, A'), what fraction of A's matched
        outgoing neighbours map to a class that A' also references?
        Score = mean across confirmed pairs.

  ambiguity_rate
        Pairs where at least one validator-style signal disagreed
        (multiple matchers proposed different B candidates for the
        same A but the engine still committed one). Reported by
        examining matcher_stats from the run.

  coverage
        a_coverage and b_coverage as fractions, copied from summary.

  overall
        A blended score in [0, 1] combining the above with hand-picked
        weights. Use it as a single dial when comparing algorithm
        variants.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .loader import load_jsonl
from .project import InMemoryProject, _stable


def _load_result(path: str) -> dict:
    with open(path) as f:
        return json.loads(f.read())


def evaluate(result: dict, A: InMemoryProject, B: InMemoryProject) -> dict:
    mapping = result["mapping"]
    a2b = {m["a"]: m["b"] for m in mapping}
    b2a = {m["b"]: m["a"] for m in mapping}

    # 1) 1-1 ratio
    one_one = sum(1 for m in mapping if b2a.get(m["b"]) == m["a"])
    one_to_one_ratio = one_one / len(mapping) if mapping else 1.0

    # 2) multi-matcher agreement
    matcher_counts = Counter(len(m["matchers"]) for m in mapping)
    mean_matchers = (sum(k * v for k, v in matcher_counts.items())
                     / max(1, sum(matcher_counts.values())))

    # 3) anchor recovery — every stable A id whose FQN exists in B
    stable_in_both = [cid for cid in A.ids()
                      if _stable(cid) and B.get(cid) is not None]
    recovered = sum(1 for cid in stable_in_both if a2b.get(cid) == cid)
    proposed_stable = sum(1 for m in mapping
                          if _stable(m["a"]) and m["a"] == m["b"])
    proposed_stable_total = sum(1 for m in mapping if _stable(m["a"]))
    precision = (proposed_stable / proposed_stable_total
                 if proposed_stable_total else 1.0)
    recall = (recovered / len(stable_in_both)
              if stable_in_both else 1.0)

    # 4) neighbour consistency
    if mapping:
        scores = []
        for m in mapping:
            a = m["a"]; b = m["b"]
            nbs = list(A.neighbours(a))
            matched_nb = [(n, a2b[n]) for n in nbs if n in a2b]
            if not matched_nb:
                continue
            b_nbs = set(B.neighbours(b))
            hit = sum(1 for _, nbb in matched_nb if nbb in b_nbs)
            scores.append(hit / len(matched_nb))
        neighbour_consistency = sum(scores) / len(scores) if scores else 0.0
    else:
        neighbour_consistency = 0.0

    # 5) ambiguity rate from matcher_stats
    stats = result.get("matcher_stats", {})
    rejected = sum(s.get("rejected", 0) for s in stats.values())
    proposed = sum(s.get("proposed", 0) for s in stats.values())
    ambiguity_rate = rejected / proposed if proposed else 0.0

    a_cov = result["summary"]["a_coverage"]
    b_cov = result["summary"]["b_coverage"]

    # Obfuscated-only coverage. The non-LX classes are 'free' anchors —
    # they round-trip by FQN and aren't really what we're trying to
    # recover. The honest progress signal is matched_obfuscated /
    # total_obfuscated, which strips the trivial denominator inflation.
    a_obf_total = sum(1 for c in A.classes() if A.is_obfuscated(c["id"]))
    b_obf_total = sum(1 for c in B.classes() if B.is_obfuscated(c["id"]))
    a_obf_matched = sum(1 for m in mapping if A.is_obfuscated(m["a"]))
    b_obf_matched = sum(1 for m in mapping if B.is_obfuscated(m["b"]))
    a_obf_cov = a_obf_matched / a_obf_total if a_obf_total else 0.0
    b_obf_cov = b_obf_matched / b_obf_total if b_obf_total else 0.0

    # Blended overall score (heuristic weights):
    #   anchor precision    : 0.30  — must not lie about anchors
    #   neighbour consistency: 0.25 — graph self-consistency
    #   anchor recall       : 0.15  — must find anchors that exist
    #   coverage (a)        : 0.20  — match as much as possible
    #   matcher agreement (norm to 0..1 on 5 matchers): 0.10
    overall = (
        0.30 * precision +
        0.25 * neighbour_consistency +
        0.15 * recall +
        0.20 * a_cov +
        0.10 * min(mean_matchers / 5.0, 1.0)
    )

    return {
        "coverage": {
            "a": a_cov, "b": b_cov,
            "a_obfuscated": a_obf_cov, "b_obfuscated": b_obf_cov,
            "a_obfuscated_matched": a_obf_matched,
            "a_obfuscated_total": a_obf_total,
        },
        "one_to_one_ratio": one_to_one_ratio,
        "anchor_recovery": {
            "stable_total": len(stable_in_both),
            "recovered": recovered,
            "precision": precision,
            "recall": recall,
        },
        "neighbour_consistency": neighbour_consistency,
        "multi_matcher_agreement": {
            "histogram": dict(matcher_counts),
            "mean_matchers_per_pair": mean_matchers,
        },
        "ambiguity_rate": ambiguity_rate,
        "overall_score": overall,
    }


def main(argv=None):
    p = argparse.ArgumentParser("apkmatch-harness")
    p.add_argument("result", help="result.json from `apkmatch.cli match`")
    p.add_argument("A", help="A.jsonl used to produce the result")
    p.add_argument("B", help="B.jsonl used to produce the result")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    print(f"[harness] loading projects...", file=sys.stderr)
    A = InMemoryProject(load_jsonl(args.A))
    B = InMemoryProject(load_jsonl(args.B))
    print(f"[harness] loading result...", file=sys.stderr)
    result = _load_result(args.result)

    report = evaluate(result, A, B)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text); f.write("\n")
    else:
        sys.stdout.write(text); sys.stdout.write("\n")


if __name__ == "__main__":
    main()
