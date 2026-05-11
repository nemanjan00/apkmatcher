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

A coverage = matched / total A classes (157538). LX coverage =
matched obfuscated / total obfuscated (140290) — the honest signal,
since non-LX classes are free anchors that round-trip by FQN.

| Iteration | A cov | LX cov | B cov | Anchor prec | Neighbour | 2+ matchers | Overall | Wall |
|---|---|---|---|---|---|---|---|---|
| matchers only | 63.4 % | — | 60.6 % | 99.3 % | 70.0 % |  2497 | 0.770 | 18 s |
| + validators (hard-veto only) | 62.4 % | — | 59.7 % | 99.3 % | 72.6 % | 2497 | 0.775 | 32 s |
| + sibling-by-mapped-super, weighted vote, sig-substituted | 63.5 % | — | 60.8 % | 99.4 % | 75.1 % | 2506 | 0.783 | 33 s |
| + call-target / field-target multisets w/ specificity scoring | 63.9 % | — | 61.1 % | 99.4 % | 75.3 % | 9082 | 0.786 | 46 s |
| + neighbours()/reverse_neighbours() bug fix (yield+return) | 66.4 % | — | 63.6 % | 99.2 % | 73.1 % | 9770 | 0.785 | 51 s |
| + EnumValueNames (tier 2) + LockStep (tier 3) | 67.0 % | — | 64.1 % | 99.2 % | 73.4 % | 9949 | 0.787 | 52 s |
| + JaccardStrings (tier 2) + looser LockStep n=3 | 67.7 % | — | 64.7 % | 99.2 % | 74.1 % | 10340 | 0.790 | 69 s |
| + ReverseLockStep + tier3_max_iters=20 | 71.7 % | — | 68.7 % | 99.2 % | 76.8 % | 10373 | 0.805 | 235 s |
| + MethodCallSetSubstituted (per-method index) | 71.9 % | 68.6 % | 68.8 % | 99.2 % | 77.3 % | 13387 | 0.807 | 396 s |
| + ExtendedByLockStep + ImplementedByLockStep | 72.7 % | 69.5 % | 69.5 % | 99.2 % | 77.4 % | 13484 | 0.809 | 403 s |
| + resource-id resolves to literal string value (was apktool placeholder) | 72.6 % | 69.4 % | 69.5 % | 99.2 % | 77.4 % | 14152 | 0.809 | 407 s |
| + LineRefMultiset (per-line stable refs) | 72.6 % | 69.3 % | 69.4 % | 99.2 % | 77.4 % | 14354 | 0.809 | 410 s |
| + negative cache (revoked pairs not re-proposable) | 72.6 % | 69.4 % | 69.4 % | 99.2 % | 77.4 % | 14365 | 0.809 | 127 s |
| + sibling_loose + WeightedNeighbourVote n=4 + n=2 | 72.7 % | 69.9 % | 69.9 % | 99.2 % | 77.4 % | 14790 | 0.811 | 152 s |
| + LockStep n=2, ReverseLockStep n=2, ImplementedBy n=2 | 76.6 % | 73.9 % | 70.6 % | 99.1 % | 78.4 % | 15481 | 0.819 | 185 s |
| + n=1 LockStep variants (single mapped neighbour) | 81.4 % | 79.2 % | 75.7 % | 99.1 % | 78.8 % | 16795 | 0.830 | 228 s |
| + cross-namespace guard (stable<->stable must FQN-match) | 81.1 % | 79.1 % | 75.6 % | 100.0 % | 79.0 % | 16777 | 0.833 | 227 s |
| + DisambiguatingLockStep | 81.4 % | 79.4 % | 75.9 % | 100.0 % | 79.1 % | 16780 | 0.833 | 259 s |
| + MethodWalk (per-method graph walk) | 83.2 % | 81.4 % | 77.8 % | 100.0 % | 76.8 % | 17233 | 0.831 | 279 s |
| + MappedNeighbourFingerprint | 84.0 % | 82.2 % | 78.6 % | 100.0 % | 77.4 % | 17641 | 0.834 | 338 s |
| + looser fingerprint pass + EnumValueNamesJaccard + NeighbourVote n=1 + positional MethodWalk | 85.2 % | 83.7 % | 80.0 % | 100.0 % | 77.1 % | 17751 | 0.836 | 338 s |
| + validator-promoted lock @ conf>=0.85 + SymmetricNeighbourFingerprint | 85.3 % | 83.7 % | 80.1 % | 100.0 % | 77.3 % | 17739 | 0.837 | 299 s |
| + Hungarian method pairing | 85.4 % | 83.8 % | 80.1 % | 100.0 % | 77.4 % | 17864 | 0.837 | 344 s |
| + AnonBodyHashMultiset (LX-stripped per-method body hashes) | 87.5 % | 86.2 % | 82.4 % | 100.0 % | 77.4 % | 20785 | 0.842 | 362 s |
| + AnonBodyHashJaccard strict (m=5, j=0.85) + (m=3, j=0.95) | 87.6 % | 86.4 % | 82.5 % | 100.0 % | 77.5 % | 21024 | 0.842 | 421 s |
| + FieldTypeSequenceSubstituted | 88.6 % | 87.4 % | 83.6 % | 100.0 % | 77.1 % | 20988 | 0.843 | 415 s |
| + FieldTypeMultiset + MethodSigSubstitutedVote | 88.6 % | 87.5 % | 84.8 % | 100.0 % | 77.1 % | 21002 | 0.843 | 424 s |
| + SiblingBySubstitutedSignatures + BestEffortContentMatch | 88.9 % | 87.8 % | 85.1 % | 100.0 % | 77.2 % | 21034 | 0.844 | 453 s |
| + two-pass substitution + DeobfuscatedFQN + method/field output | 89.3 % | 88.2 % | 85.4 % | 100.0 % | 76.2 % | 21167 | 0.842 | 460 s |
| + full-sig MethodWalk + sig-position LX inference | 89.5 % | 88.5 % | 85.6 % | 100.0 % | 76.0 % | 21218 | 0.843 | 512 s |
| + FieldPositionLXVote + ImplsPositionLXVote + SuperLXVote | 90.9 % | 90.0 % | 87.0 % | 100.0 % | 76.1 % | 21433 | 0.845 | 519 s |
| + AnnsPositionLXVote + n=1 variants | 91.8 % | 91.0 % | 87.8 % | 100.0 % | 75.7 % | 21610 | 0.846 | 516 s |
| + BodyLXSequenceVote (per-method body LX-sequence position vote) | 92.6 % | 92.0 % | 88.6 % | 100.0 % | 75.0 % | 21672 | 0.846 | 534 s |
| + neighbour-consistency cleanup pass + UniqueString conf=0.95 | 92.3 % | 91.6 % | 88.4 % | 100.0 % | 76.8 % | 21675 | 0.850 | 510 s |
| + post-two-pass cleanup pass | 91.6 % | 90.9 % | 87.7 % | 100.0 % | 77.8 % | 21664 | 0.851 | 583 s |
| + second tier-3 sweep after first cleanup | 92.4 % | 91.7 % | 88.5 % | 100.0 % | 77.8 % | 21695 | 0.852 | 742 s |
| + cleanup min_mapped 3 → 2 (both passes) | **92.1 %** | **91.4 %** | **88.2 %** | **100.0 %** | **78.7 %** | **21694** | **0.854** | 821 s |

## Status

Working prototype. See `DESIGN.md` for the full architecture,
including the deferred-question / reactive-invalidation model that
validators are eventually meant to support.
