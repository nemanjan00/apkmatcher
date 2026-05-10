# Project conventions

## Layout

- `apkmatch/` — pure library, importable. Never imports from `cli.py` or
  `harness.py`.
- `apkmatch/cli.py` — `python -m apkmatch.cli`. JSON in/out.
- `apkmatch/harness.py` — `python -m apkmatch.harness`. Self-evaluation.
- `apkmatch/interfaces.py` / `model.py` — design contract. Concrete code
  satisfies these structurally (Protocol-typed), not by inheritance.
- `samples/` — apktool-decoded test fixtures. Gitignored.
- `build/` — JSONL indexes, result.json, etc. Gitignored.

## Three concepts

Don't blur them:

1. **Neighbours** — typed edges between classes (inheritance, calls,
   field access, …). Built once at load.
2. **Matchers** — propose `(a, b, confidence, matcher_id)` candidates.
   They are STATELESS w.r.t. iteration; the engine schedules them by
   tier.
3. **Candidate validators** — score a `(a, b)` pair on a 1–10 scale.
   They consume read-only `ProjectView`s and a `MappingReader` (which
   records lookups for reactive invalidation). They NEVER mutate
   anything.

## Scoring

- Validators return `Score(value: int 1..10, provisional: bool, deps: tuple[ClassId])`.
- `value` is ALWAYS a real opinion based on what was knowable, never a
  "I don't know" placeholder.
- `provisional=True` means "the score would change with more data"
  (some dep is currently unmapped). It drives RE-EVALUATION, not
  commitment.
- `1` is a hard veto — kills the candidate regardless of other scores.

## Reliability rules

- Anchor precision is more important than coverage. Don't trade away
  precision for a couple of percentage points of recall.
- Provisional validator scores must not raise confidence above a
  non-provisional matcher's reading.
- Tier-1 matches lock and are immune to displacement / revocation.
- Use `_log("...")` for per-step engine progress so long runs aren't
  opaque.

## Telemetry in every commit

Every commit that touches matchers, validators, or the engine MUST
include before/after coverage and self-eval numbers in the commit
message. The README table tracks the canonical baseline; update it
when a change moves the needle. Use Instagram 415.0.0 vs 416.0.0 as
the reference workload.

Format (from `python -m apkmatch.harness`):

```
| iteration | A cov | B cov | anchor prec | neighbour | overall | wall |
```

## Performance budget

The full Instagram pipeline runs in seconds-to-tens-of-seconds. If a
change pushes it past a minute without a coverage justification,
profile before committing.

## Don't

- Don't add throwaway scripts in `tools/` or similar — extend the
  library or add a subcommand to the CLI.
- Don't read the mapping directly from a validator — go through
  `MappingReader` so the dep graph stays correct.
- Don't compute the same similarity metric in two different
  matchers/validators — promote it to `Oracle` (when wired) or to
  `project.py`.
- Don't widen `validator_revoke_threshold` without measuring — it
  oscillated tier-3 the first time we tried.
