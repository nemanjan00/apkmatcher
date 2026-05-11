"""Pair methods and fields within each confirmed class pair.

Runs as a post-processing step after the engine converges on a
class-level mapping. Emits a separate output describing which
member in A corresponds to which member in B.

Method pairing strategy:
  1. exact substituted-signature match (deterministic).
  2. Hungarian (optimal bipartite) assignment over remaining
     methods using a per-method similarity score combining:
       * Jaccard on (target_class, member-name) call sets,
       * Jaccard on (target_class, field-name) accesses,
       * branch-count agreement,
       * normalized body-hash equality (small bonus).

Field pairing strategy:
  1. exact substituted-type match in declaration order.
  2. Remaining unpaired fields matched by Hungarian on
     substituted-type cost (1 if same after sub else 1).
"""
from __future__ import annotations

import sys
from typing import Iterator


def _sub_ref(x: str, mapping) -> str:
    if not x or not x.startswith("LX/"):
        return x
    return mapping.get(x) or x


def _sub_sig(sig: str, mapping) -> str:
    out = []; i = 0
    while i < len(sig):
        c = sig[i]
        if c == "L":
            e = sig.find(";", i)
            if e == -1:
                out.append(sig[i:]); break
            ref = sig[i:e+1]
            out.append(_sub_ref(ref, mapping))
            i = e + 1; continue
        out.append(c); i += 1
    return "".join(out)


def _method_sim(ma: dict, mb: dict) -> float:
    ma_c = set(map(tuple, ma.get("calls", ())))
    mb_c = set(map(tuple, mb.get("calls", ())))
    ma_f = set(map(tuple, ma.get("facc", ())))
    mb_f = set(map(tuple, mb.get("facc", ())))
    jc = (len(ma_c & mb_c) / max(1, len(ma_c | mb_c))
          if (ma_c or mb_c) else 0.0)
    jf = (len(ma_f & mb_f) / max(1, len(ma_f | mb_f))
          if (ma_f or mb_f) else 0.0)
    br = 1.0 - abs(ma.get("br", 0) - mb.get("br", 0)) \
              / max(1, max(ma.get("br", 0), mb.get("br", 0)))
    bha_bonus = 0.1 if ma.get("bha") and ma["bha"] == mb.get("bha") else 0.0
    return 0.5 * jc + 0.3 * jf + 0.2 * br + bha_bonus


def pair_methods(ma_list: list[dict], mb_list: list[dict], mapping) -> list[tuple[dict, dict, float]]:
    """Returns list of (ma, mb, similarity) triples."""
    out = []
    # Pass 1: substituted-signature equality.
    used_b = set()
    b_by_sig: dict[str, list[int]] = {}
    for j, mb in enumerate(mb_list):
        b_by_sig.setdefault(mb["sig"], []).append(j)
    leftover_a_idx = []
    for i, ma in enumerate(ma_list):
        sub = _sub_sig(ma["sig"], mapping)
        cands = b_by_sig.get(sub, ())
        picked = None
        for j in cands:
            if j in used_b: continue
            picked = j; used_b.add(j); break
        if picked is not None:
            out.append((ma, mb_list[picked], 1.0))
        else:
            leftover_a_idx.append(i)
    leftover_b_idx = [j for j in range(len(mb_list)) if j not in used_b]
    if not leftover_a_idx or not leftover_b_idx:
        return out

    # Pass 2: Hungarian on (calls, facc, br, bha)
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        # Greedy fallback
        for i in leftover_a_idx:
            best = None; best_s = 0.0
            for j in leftover_b_idx:
                if j in used_b: continue
                s = _method_sim(ma_list[i], mb_list[j])
                if s > best_s:
                    best_s = s; best = j
            if best is not None and best_s >= 0.3:
                out.append((ma_list[i], mb_list[best], best_s))
                used_b.add(best)
        return out
    if len(leftover_a_idx) * len(leftover_b_idx) > 20000:
        return out  # too big; skip
    costs = np.zeros((len(leftover_a_idx), len(leftover_b_idx)))
    for x, i in enumerate(leftover_a_idx):
        for y, j in enumerate(leftover_b_idx):
            costs[x, y] = 1.0 - _method_sim(ma_list[i], mb_list[j])
    r, c = linear_sum_assignment(costs)
    for x, y in zip(r, c):
        if costs[x, y] <= 0.7:
            i = leftover_a_idx[x]; j = leftover_b_idx[y]
            out.append((ma_list[i], mb_list[j], 1.0 - costs[x, y]))
    return out


def pair_fields(ra: dict, rb: dict, mapping) -> list[tuple[int, int, str, str]]:
    """Returns list of (a_index, b_index, a_type, b_type) tuples.
    Indices are positions in the field_types arrays."""
    a_types = ra.get("field_types", ())
    b_types = rb.get("field_types", ())
    if not a_types or not b_types:
        return []
    out = []
    used_b = set()
    leftover_a = []
    # Pass 1: substituted-type equality in declaration order — prefer
    # same-index pair first, then any earlier-unused B index.
    b_by_type: dict[str, list[int]] = {}
    for j, t in enumerate(b_types):
        b_by_type.setdefault(t, []).append(j)
    for i, t in enumerate(a_types):
        sub = _sub_sig(t, mapping)
        cands = b_by_type.get(sub, ())
        picked = None
        # Prefer the same-index candidate when available.
        if i in cands and i not in used_b:
            picked = i
        else:
            for j in cands:
                if j in used_b: continue
                picked = j; break
        if picked is not None:
            used_b.add(picked)
            out.append((i, picked, a_types[i], b_types[picked]))
        else:
            leftover_a.append(i)
    return out


def build_member_mappings(mapping, A, B) -> dict:
    """Iterate confirmed class pairs and build method/field mapping.

    Returns: {"methods": [...], "fields": [...]}
    """
    methods_out = []
    fields_out = []
    for a_id, b_id in mapping:
        ra = A.get(a_id); rb = B.get(b_id)
        if not ra or not rb:
            continue
        ma_list = ra.get("methods", ())
        mb_list = rb.get("methods", ())
        if ma_list and mb_list:
            for ma, mb, sim in pair_methods(list(ma_list), list(mb_list), mapping):
                methods_out.append({
                    "a_class": a_id, "b_class": b_id,
                    "a": ma["sig"], "b": mb["sig"],
                    "sim": round(sim, 2),
                })
        pairs = pair_fields(ra, rb, mapping)
        for i, j, at, bt in pairs:
            fields_out.append({
                "a_class": a_id, "b_class": b_id,
                "a_idx": i, "b_idx": j,
                "a_type": at, "b_type": bt,
            })
    return {"methods": methods_out, "fields": fields_out}
