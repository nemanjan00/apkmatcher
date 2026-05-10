# apkmatch

Match obfuscated classes across two versions of the same Android app.

R8 rotates class names every build (e.g. Instagram's `LX/8WN;` from one
release means something entirely different in the next), but the
**content** of a class — its strings, the framework classes it touches,
its native methods, its position in the call graph — is mostly stable.
`apkmatch` fuses many of those signals to recover the
`old_name → new_name` mapping with provenance.

## Three surfaces

| Surface | Module | Purpose |
|---|---|---|
| Library | `apkmatch/` | Pure Python; `loader`, `project`, `mapping`, `matchers`, `validators`, `engine` |
| CLI | `apkmatch.cli` | `index` / `match` / `run` subcommands, JSON in and out |
| Test harness | `apkmatch.harness` | Self-evaluates a result (no external ground truth required) |

The library doesn't know either front-end exists. The CLI and the
harness import the library. Add a new tool by importing it.

## Quick start

```bash
# 1. apktool d both APKs somewhere (left as exercise — apktool is heavy).
#    Each tree should contain smali/, smali_classesN/, res/values/public.xml.

# 2. Index each tree (parses smali into one JSON object per class).
python -m apkmatch.cli index path/to/old.apktool build/old.jsonl
python -m apkmatch.cli index path/to/new.apktool build/new.jsonl

# 3. Run the matcher.
python -m apkmatch.cli match build/old.jsonl build/new.jsonl --out build/result.json

# 4. Self-evaluate.
python -m apkmatch.harness build/result.json build/old.jsonl build/new.jsonl
```

`run` does steps 2 and 3 together with caching.

## How it works

1. **Indexing.** Each smali tree is parsed once into a JSONL of class
   records: super, interfaces, modifiers, strings, native method
   symbols, annotations, method signatures, normalized body hashes,
   outgoing call/access/type-ref sets, and resolved resource-ID
   constants (from `res/values/public.xml`).

2. **Matchers run tier by tier.** Each matcher proposes
   `(a_class, b_class, confidence)` candidates from an inverted index
   over the two projects.

   - *Tier 1 (anchors, lockable):* `fqn_stable` (non-rotating
     namespaces), `native_syms` (JNI symbol sets), `identical_strings`
     (≥3 strings, ≥30 chars total), `long_unique_string` (single
     ≥20-char string globally unique).
   - *Tier 2 (content fingerprints):* `unique_string`, `string_pair`,
     `stringset_hash`, `strings_plus_refs`, `stable_refs_ms`.
   - *Tier 3 (propagation, iterated):* `neighbour_vote` — for each
     unmatched A class, propose the B class that the most of its
     already-matched neighbours' partners reverse-reference.

3. **Validators score every committed pair.** Three so far:
   `shape` (modifier and size compatibility, can hard-veto),
   `signature_refs` (substitute matched class refs into A's method
   signatures, check overlap with B's), `neighbour_consistency`
   (fraction of A's matched outgoing neighbours that land inside B's
   neighbour set). Each returns an integer 1–10 plus a `provisional`
   flag (= "score would change with more data") and the set of
   mapping keys it consulted (for reactive re-evaluation).

4. **Mapping is non-monotonic across epochs.** A confirmed pair can be
   displaced by a higher-confidence candidate (modulo a displacement
   margin) or revoked by a hard veto. Locked pairs from tier-1 anchors
   are immune.

5. **Convergence.** Tier-3 iterates until K consecutive sweeps add
   fewer than `min_churn` pairs.

## Reliability principles

- **Anchor precision over coverage.** A few thousand confidently wrong
  matches contaminate every downstream signal (neighbour vote,
  signature substitution). The engine prefers to leave a class
  unmatched than to commit a low-evidence guess.
- **Stable namespaces are anchors.** Non-`LX/` classes (framework,
  Kotlin runtime, OSS libs, kept packages) round-trip by FQN.
- **Identical-string sets are anchors.** Two classes sharing the same
  set of distinct ≥4-char strings, with ≥30 chars pooled length, are
  effectively the same class.
- **Provisional scores never raise confidence.** A validator that says
  "I'm guessing 8 based on partial data" cannot promote a pair above
  what a non-provisional matcher already gave it. Provisional scores
  drive re-evaluation, not commitment.

## Telemetry

Every commit to this repo documents before/after measurements so
regressions are catchable in `git log`. Baseline on Instagram
415.0.0 vs 416.0.0 (157538 vs 164620 classes):

| Iteration | A cov | B cov | Anchor prec | Neighbour | 2+ matchers | Overall | Wall |
|---|---|---|---|---|---|---|---|
| matchers only | 63.4 % | 60.6 % | 99.3 % | 70.0 % |  2497 | 0.770 | 18 s |
| + validators (hard-veto only) | 62.4 % | 59.7 % | 99.3 % | 72.6 % | 2497 | 0.775 | 32 s |
| + sibling-by-mapped-super, weighted vote, sig-substituted | 63.5 % | 60.8 % | 99.4 % | 75.1 % | 2506 | 0.783 | 33 s |
| + call-target / field-target multisets w/ specificity scoring | 63.9 % | 61.1 % | 99.4 % | 75.3 % | 9082 | 0.786 | 46 s |
| + neighbours()/reverse_neighbours() bug fix (yield+return) | 66.4 % | 63.6 % | 99.2 % | 73.1 % | 9770 | 0.785 | 51 s |
| + EnumValueNames (tier 2) + LockStep (tier 3) | 67.0 % | 64.1 % | 99.2 % | 73.4 % | 9949 | 0.787 | 52 s |
| + JaccardStrings (tier 2) + looser LockStep n=3 | 67.7 % | 64.7 % | 99.2 % | 74.1 % | 10340 | 0.790 | 69 s |
| + ReverseLockStep + tier3_max_iters=20 | **71.7 %** | **68.7 %** | 99.2 % | **76.8 %** | 10373 | **0.805** | 235 s |

## Status

Working prototype. See `DESIGN.md` for the full architecture,
including the deferred-question / reactive-invalidation model that
validators are eventually meant to support.
