"""Load an apktool-decoded tree into the in-memory model.

Single-pass smali parser. Emits one record per class with all the
fields needed by matchers, validators, and the oracle. The parser is
deliberately line-prefix based — no regex backtracking — so it streams
through a 1.4 GB Instagram smali tree in a few seconds on 8 cores.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Iterable

# Per-worker resource map. Populated by _init_worker and read by parse_class.
_RES_MAP: dict[int, str] = {}


def _init_worker(res_map: dict[int, str]) -> None:
    global _RES_MAP
    _RES_MAP = res_map


def load_resource_map(apktool_root: str) -> dict[int, str]:
    """Parse res/values/public.xml into {int_id: 'type/name'}.

    Resource IDs in smali appear as integer constants (e.g.
    `const v0, 0x7f0a01b2`). Resolving them to symbolic names like
    'id/profile_button' adds a huge cross-version-stable signal —
    resource names rarely change even when class names rotate.
    """
    import xml.etree.ElementTree as ET
    path = os.path.join(apktool_root, "res", "values", "public.xml")
    if not os.path.exists(path):
        return {}
    out: dict[int, str] = {}
    for el in ET.parse(path).getroot().iter("public"):
        try:
            rid = int(el.attrib["id"], 16)
        except (KeyError, ValueError):
            continue
        out[rid] = f"R.{el.attrib['type']}.{el.attrib['name']}"
    return out

# --------------------------------------------------------------------------- #
# Smali parser
# --------------------------------------------------------------------------- #

P_CLASS  = ".class "
P_SUPER  = ".super "
P_IMPL   = ".implements "
P_FIELD  = ".field "
P_METHOD = ".method "
P_END_M  = ".end method"
P_ANN    = ".annotation"
P_CSTR   = "const-string"
P_INVOKE = "invoke-"
P_NEW    = "new-instance "
P_CHECK  = "check-cast "
P_INSTO  = "instance-of "

_MODS = {"public", "private", "protected", "static", "final",
         "abstract", "interface", "enum", "annotation", "synthetic",
         "native", "bridge", "varargs"}


def _normalize_op(line: str) -> str:
    """Strip register operands and labels so the same instruction in two
    builds hashes to the same value when only register allocation differs."""
    parts = line.split(None, 1)
    if not parts:
        return ""
    op = parts[0]
    if len(parts) == 1:
        return op
    out = [op]
    for tok in parts[1].split(", "):
        tok = tok.strip()
        if not tok:
            continue
        if tok[0] in "vp" and tok[1:].isdigit():
            continue
        if tok.startswith(":"):
            continue
        if tok.startswith("{") and tok.endswith("}"):
            continue
        out.append(tok)
    return " ".join(out)


def _maybe_resource(line: str) -> str | None:
    """If `line` is `const vN, 0x7f...`, return the resolved resource
    name; otherwise None."""
    if not (line.startswith("const ") or line.startswith("const/4 ")
            or line.startswith("const/16 ")):
        return None
    parts = line.rsplit(", ", 1)
    if len(parts) != 2:
        return None
    val = parts[1].strip()
    if not (val.startswith("0x7f") or val.startswith("0x01")):
        return None  # 0x7f = app, 0x01 = framework
    try:
        n = int(val, 16)
    except ValueError:
        return None
    return _RES_MAP.get(n)


def parse_class(path: str, bucket: str) -> dict | None:
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    cid = None
    super_ = None
    impls: list[str] = []
    mods: set[str] = set()
    nf = nm = ns = nn = 0
    strings: list[str] = []
    anns: list[str] = []
    sigs: list[str] = []
    body_hashes: list[str] = []
    calls: set[str] = set()
    facc: set[str] = set()
    trefs: set[str] = set()
    native_syms: list[str] = []
    # Method-level callsites & field accesses: (target_class, member_name)
    # tuples. Captures "who do I call" at the method-name granularity,
    # which is much more discriminating than just "which class". Member
    # names that are themselves obfuscated (single-letter / LX-style)
    # are kept verbatim — the next-version-equivalent pair is just as
    # likely to use the same rotated names on both sides if the call
    # target is an LX class that survives unchanged.
    call_targets: list[tuple[str, str]] = []
    field_targets: list[tuple[str, str]] = []
    # Per-method records: list of dicts with method's own info plus
    # the call/field targets it specifically performed. Lets matchers
    # reason about method pairing within paired classes
    # ("X.foo calls Y.bar; if Y maps to Y' then X.foo' should call Y'.bar'").
    methods: list[dict] = []
    cur_calls: list[tuple[str, str]] = []
    cur_facc: list[tuple[str, str]] = []
    cur_strings: list[str] = []
    cur_n_branches = 0

    in_method = False
    cur_body: list[str] = []
    cur_is_native = False
    cur_method_name = ""
    cur_sig = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue

        if in_method:
            if line.startswith(P_END_M):
                if cur_body:
                    h = hashlib.blake2b(
                        "\n".join(sorted(cur_body)).encode(), digest_size=8
                    ).hexdigest()
                    body_hashes.append(h)
                else:
                    h = ""
                methods.append({
                    "sig": cur_sig, "name": cur_method_name,
                    "native": cur_is_native, "bh": h,
                    "calls": cur_calls, "facc": cur_facc,
                    "strs": cur_strings, "br": cur_n_branches,
                })
                in_method = False
                cur_body = []; cur_calls = []; cur_facc = []
                cur_strings = []; cur_n_branches = 0
                continue
            if line.startswith(P_CSTR):
                q1 = line.find('"'); q2 = line.rfind('"')
                if q1 != -1 and q2 > q1:
                    strings.append(line[q1+1:q2])
                    cur_strings.append(line[q1+1:q2])
                continue
            if line.startswith(P_INVOKE):
                arrow = line.find("->")
                if arrow != -1:
                    sp = line.rfind(" ", 0, arrow)
                    if sp != -1:
                        cls = line[sp+1:arrow]
                        if cls.startswith("L") and cls.endswith(";"):
                            calls.add(cls)
                            paren = line.find("(", arrow + 2)
                            if paren != -1:
                                mname = line[arrow+2:paren]
                                call_targets.append((cls, mname))
                                cur_calls.append((cls, mname))
                cur_body.append(_normalize_op(line))
                continue
            if line[:4] in ("iget", "iput", "sget", "sput"):
                arrow = line.find("->")
                if arrow != -1:
                    sp = line.rfind(" ", 0, arrow)
                    if sp != -1:
                        cls = line[sp+1:arrow]
                        if cls.startswith("L") and cls.endswith(";"):
                            facc.add(cls)
                            colon = line.find(":", arrow + 2)
                            if colon != -1:
                                fname = line[arrow+2:colon]
                                field_targets.append((cls, fname))
                                cur_facc.append((cls, fname))
                cur_body.append(_normalize_op(line))
                continue
            if line.startswith("if-") or line.startswith("packed-switch") \
                    or line.startswith("sparse-switch") or line.startswith("goto"):
                cur_n_branches += 1
                cur_body.append(_normalize_op(line)); continue
            if (line.startswith(P_NEW) or line.startswith(P_CHECK)
                    or line.startswith(P_INSTO)):
                last = line.rsplit(None, 1)[-1]
                if last.startswith("L") and last.endswith(";"):
                    trefs.add(last)
                cur_body.append(_normalize_op(line))
                continue
            # Resolve resource-id constants if the body has them.
            res = _maybe_resource(line)
            if res:
                strings.append(res)
            cur_body.append(_normalize_op(line))
            continue

        if line.startswith(P_CLASS):
            toks = line.split()
            cid = toks[-1]
            for t in toks[1:-1]:
                if t in _MODS:
                    mods.add(t)
            continue
        if line.startswith(P_SUPER):
            super_ = line.split()[-1]; continue
        if line.startswith(P_IMPL):
            impls.append(line.split()[-1]); continue
        if line.startswith(P_FIELD):
            nf += 1
            tail = line[len(P_FIELD):]
            if " static " in (" " + tail + " "):
                ns += 1
            colon = tail.rfind(":")
            if colon != -1:
                ftype = tail[colon+1:].split()[0]
                if ftype.startswith("L") and ftype.endswith(";"):
                    trefs.add(ftype)
            continue
        if line.startswith(P_METHOD):
            nm += 1
            in_method = True
            cur_body = []
            cur_calls = []; cur_facc = []; cur_strings = []
            cur_n_branches = 0
            tail = line[len(P_METHOD):]
            cur_is_native = " native " in (" " + tail + " ")
            sig = tail.split()[-1]
            cur_sig = sig
            sigs.append(sig)
            cur_method_name = sig.split("(", 1)[0]
            if cur_is_native:
                nn += 1
                native_syms.append(cur_method_name)
                methods.append({
                    "sig": sig, "name": cur_method_name, "native": True,
                    "bh": "", "calls": [], "facc": [], "strs": [], "br": 0,
                })
                in_method = False  # native methods have no body
            # Pull type refs out of the signature.
            lp = sig.find("("); rp = sig.find(")", lp)
            if lp != -1 and rp != -1:
                params = sig[lp+1:rp]
                i = 0
                while i < len(params):
                    c = params[i]
                    if c == "L":
                        e = params.find(";", i)
                        if e != -1:
                            trefs.add(params[i:e+1]); i = e + 1; continue
                    if c == "[":
                        i += 1; continue
                    i += 1
                ret = sig[rp+1:]
                if ret.startswith("L") and ret.endswith(";"):
                    trefs.add(ret)
            continue
        if line.startswith(P_ANN):
            for tok in line.split():
                if tok.startswith("L") and tok.endswith(";"):
                    anns.append(tok); break
            continue

    if cid is None:
        return None
    calls.discard(cid); facc.discard(cid); trefs.discard(cid)
    return {
        "id": cid, "super": super_, "impls": impls,
        "mods": sorted(mods),
        "nf": nf, "nm": nm, "ns": ns, "nn": nn,
        "strings": strings, "anns": anns,
        "bucket": bucket, "sigs": sigs, "bh": body_hashes,
        "calls": sorted(calls), "facc": sorted(facc), "trefs": sorted(trefs),
        "native_syms": native_syms,
        "call_targets": call_targets,
        "field_targets": field_targets,
        "methods": methods,  # per-method records (sig, calls, facc, strs, br, bh)
    }


def _worker(args):
    return parse_class(*args)


def iter_class_files(root: str) -> Iterable[tuple[str, str]]:
    """Yield (path, bucket) tuples for every .smali under apktool roots."""
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("smali"):
            continue
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        for dp, _, files in os.walk(sub):
            for fn in files:
                if fn.endswith(".smali"):
                    yield os.path.join(dp, fn), entry


def index_tree(root: str, out_path: str, workers: int = 8,
               quiet: bool = False) -> int:
    jobs = list(iter_class_files(root))
    res_map = load_resource_map(root)
    if not quiet:
        print(f"[loader] {len(jobs)} smali files under {root} "
              f"({len(res_map)} resource ids)", file=sys.stderr)
    t0 = time.time(); n = 0
    with open(out_path, "w") as out, Pool(
        workers, initializer=_init_worker, initargs=(res_map,)
    ) as pool:
        for rec in pool.imap_unordered(_worker, jobs, chunksize=200):
            if rec is None:
                continue
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
    if not quiet:
        print(f"[loader] indexed {n} classes in {time.time()-t0:.1f}s "
              f"-> {out_path}", file=sys.stderr)
    return n


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]
