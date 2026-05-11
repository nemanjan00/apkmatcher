"""CLI — `python -m apkmatch.cli`.

Three subcommands:
  index <apktool-tree> <out.jsonl>
        Parse smali, emit one JSON object per class.

  match <A.jsonl> <B.jsonl> [--out result.json]
        Run the engine, emit JSON: {mapping, matcher_stats, summary}.

  run   <A-tree> <B-tree> [--out result.json] [--cache-dir DIR]
        Index both trees (cached) then match. Convenience wrapper.

All output goes to stdout unless --out is given. Nothing else is
printed to stdout — progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .engine import Engine
from .loader import index_tree, load_jsonl
from .member_pairing import build_member_mappings
from .project import InMemoryProject


def _emit(payload, out_path: str | None) -> None:
    text = json.dumps(payload, separators=(",", ":"))
    if out_path:
        with open(out_path, "w") as f:
            f.write(text); f.write("\n")
    else:
        sys.stdout.write(text); sys.stdout.write("\n")


def cmd_index(args):
    index_tree(args.tree, args.out)


def _build_result(eng: Engine, run, t_load: float, t_run: float):
    mapping = []
    for a, b in eng.mapping:
        mapping.append({
            "a": a, "b": b,
            "confidence": eng.mapping.confidence(a),
            "matchers": eng.mapping.matchers_for(a, b),
            "locked": eng.mapping.is_locked(a),
        })
    return {
        "summary": {
            "a_total": run.a_total,
            "b_total": run.b_total,
            "matched": run.final_size,
            "a_coverage": run.final_size / run.a_total if run.a_total else 0.0,
            "b_coverage": run.final_size / run.b_total if run.b_total else 0.0,
            "epochs": run.epochs,
            "converged": run.converged,
            "load_seconds": round(t_load, 2),
            "run_seconds": round(t_run, 2),
        },
        "matcher_stats": run.matcher_stats,
        "validator_stats": run.validator_stats or {},
        "mapping": mapping,
    }


def cmd_match(args):
    t0 = time.time()
    print(f"[cli] loading {args.A} and {args.B}", file=sys.stderr)
    A = InMemoryProject(load_jsonl(args.A))
    B = InMemoryProject(load_jsonl(args.B))
    t_load = time.time() - t0
    print(f"[cli] |A|={len(A)} |B|={len(B)}  load={t_load:.1f}s", file=sys.stderr)

    t1 = time.time()
    eng = Engine(A, B)
    run = eng.run()
    t_run = time.time() - t1
    print(f"[cli] matched {run.final_size} pairs in {t_run:.1f}s "
          f"(epochs={run.epochs}, converged={run.converged})", file=sys.stderr)

    print(f"[cli] building method/field mappings...", file=sys.stderr)
    t2 = time.time()
    members = build_member_mappings(eng.mapping, A, B)
    print(f"[cli]   methods: {len(members['methods'])}, "
          f"fields: {len(members['fields'])}  ({time.time()-t2:.1f}s)",
          file=sys.stderr)

    result = _build_result(eng, run, t_load, t_run)
    result["method_mapping"] = members["methods"]
    result["field_mapping"] = members["fields"]
    _emit(result, args.out)


def cmd_run(args):
    cache = args.cache_dir or "build"
    os.makedirs(cache, exist_ok=True)
    a_jsonl = os.path.join(cache, "A.jsonl")
    b_jsonl = os.path.join(cache, "B.jsonl")
    if not os.path.exists(a_jsonl) or args.force_index:
        index_tree(args.A_tree, a_jsonl)
    else:
        print(f"[cli] using cached {a_jsonl}", file=sys.stderr)
    if not os.path.exists(b_jsonl) or args.force_index:
        index_tree(args.B_tree, b_jsonl)
    else:
        print(f"[cli] using cached {b_jsonl}", file=sys.stderr)

    class _A: pass
    a = _A(); a.A = a_jsonl; a.B = b_jsonl; a.out = args.out
    cmd_match(a)


def main(argv=None):
    p = argparse.ArgumentParser("apkmatch")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="parse an apktool tree into JSONL")
    p_idx.add_argument("tree")
    p_idx.add_argument("out")
    p_idx.set_defaults(func=cmd_index)

    p_m = sub.add_parser("match", help="match two indexed projects")
    p_m.add_argument("A"); p_m.add_argument("B")
    p_m.add_argument("--out", default=None)
    p_m.set_defaults(func=cmd_match)

    p_r = sub.add_parser("run", help="index + match (with caching)")
    p_r.add_argument("A_tree"); p_r.add_argument("B_tree")
    p_r.add_argument("--out", default=None)
    p_r.add_argument("--cache-dir", default=None)
    p_r.add_argument("--force-index", action="store_true")
    p_r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
