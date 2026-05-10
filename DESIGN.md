# APK Class Matcher — Design

A tool that, given two APK versions of the same app, produces a mapping from
obfuscated class names in version A to their counterparts in version B. The
target use case is tracking Proguard/R8-obfuscated classes across releases of
apps like Instagram, where names like `X.0Aa` rotate every build but the
underlying class identity is stable.

## Problem statement

Obfuscation rotates class names (and member names) but preserves a great deal
of structural and content evidence:

- Strings, resource IDs, native symbols, manifest entries.
- Inheritance hierarchies and interface implementations.
- The shape of the call/reference graph between classes.
- Bytecode of method bodies (modulo register renaming).

The job is to fuse these signals into a stable old→new mapping with a
confidence score, even though no single signal is reliable in isolation:

- Strings change as features ship and copy is reworded.
- Class shape changes as code is added/removed.
- Even the call graph mutates — methods get inlined, split, or moved.

## Three-concept architecture

The pipeline is built around three orthogonal concepts. Each is a plug-in
point — new matchers, validators, and neighbour relations can be added
without touching the others.

### 1. Neighbours — the class-relationship graph

Both APKs are reduced to a directed labelled multigraph:

- **Nodes** = classes (one graph per APK).
- **Edges** = typed references from one class to another. The edge type is
  the kind of relationship (inheritance, field type, method call, …).

Edge types, ordered by how strongly they pin identity:

1. **Inheritance** — `extends` (one outgoing per class) and `implements`
   (one outgoing per declared interface). Near-deterministic when the
   target is a framework class or already-matched.
2. **Enclosing class** — for inner / anonymous / synthetic classes. The
   `A$N` clusters are usually a fully-determined unit once the outer is
   matched.
3. **Field types** — declared types of instance + static fields.
4. **Method signatures** — parameter and return types.
5. **Method calls** — invokevirtual/static/interface/direct targets.
6. **Field access** — getfield/getstatic/putfield/putstatic targets.
7. **Annotations** — `@B` on the class or any member; values that are
   class literals or contain class names.
8. **Generic args** — `List<B>`, `Map<K,B>`, etc., parsed from the generic
   signature attribute.
9. **Exceptions** — thrown / caught.
10. **Cast / instanceof targets**.
11. **Class literal references** — `B.class`, `Class.forName("...")` with
    string constants.

Each edge carries a weight reflecting how strong the identity signal is
when both endpoints match. Inheritance and enclosing edges are weighted
roughly an order of magnitude above call/access edges.

The graph is built once per APK and is the substrate every other component
queries.

### 2. Matchers — propose candidate pairings

A matcher consumes one or both graphs and emits `(class_in_A, class_in_B,
matcher_id, evidence)` tuples. Matchers run in a fixed order from most to
least reliable; later matchers see the candidates and confirmed matches
emitted by earlier ones, so propagation effects compound.

**Tier 1 — anchors (near-certain, no propagation needed):**

- **Kept FQN** — class name survives obfuscation (kept by Proguard rules,
  Android framework subclasses, classes referenced from the manifest).
- **Native method symbols** — JNI symbols `Java_pkg_Cls_method` are baked
  into `.so` files and rarely rotated. Classes with native methods get
  pinned by symbol name.
- **Manifest components** — `<activity>`, `<service>`, `<receiver>`,
  `<provider>` entries with `android:name`.
- **Resource IDs in code** — references to `R$id`, `R$layout`, etc.
  Public resource names are stable; classes that touch a unique R-id are
  cheaply pinnable.
- **Serialization fingerprints** — `serialVersionUID` value, `Parcelable`
  `CREATOR` field shape, `@Keep` classes, GSON `TypeAdapter`s.

**Tier 2 — content fingerprints:**

- **Unique string literal** — a string that appears in exactly one class
  in A and exactly one in B. Trivially pairs them.
- **Rare-string set overlap** — for classes with no globally unique
  string, score by Jaccard of strings that are rare in both APKs (TF-IDF
  over class-as-document).
- **Constant tuple** — a tuple of unusual numeric/long constants.
- **Annotation values** — `@SerializedName("foo")`, Retrofit
  `@GET("/v1/whatever")`, OkHttp header names. The values are user-facing
  and rarely change.
- **Enum value names** — enums often keep their `name()` strings even
  when the enclosing class is renamed (reflection, JSON).
- **Normalized bytecode hash** — strip register numbers and local labels,
  keep opcodes plus resolved framework refs, hash per method. Match
  classes whose bytecode-hash multisets agree.

**Tier 3 — structural / propagation (run after tiers 1–2 seed the graph):**

- **Neighbour vote** — a candidate (A, A') is confirmed when a threshold
  fraction of A's already-matched neighbours map to A''s neighbours under
  the current mapping.
- **Lock-step** — when N−1 of a class's outgoing edges are already
  matched, the Nth is forced if the candidate has the matching outgoing
  shape.
- **Signature substitution** — substitute matched class refs into A's
  method signatures and look for an A' with the same multiset of
  substituted signatures.
- **Small-clique isomorphism** — inner-class clusters, sealed
  hierarchies, and `enum` value-classes form small subgraphs that are
  often isomorphic. Match the whole clique at once.
- **Bytecode similarity with substitution** — once class refs in
  bytecode can be rewritten via the current mapping, do a real bytecode
  diff (not just hash equality) for fuzzy method-pair matching.

Tier 3 is iterated to a fixpoint: each new confirmed match in the
mapping unlocks more neighbour-vote and lock-step matches.

### 3. Candidate validators — score a proposed pair

A validator takes a `(A, A')` candidate plus the current mapping state
and returns an integer score on a **1–10 scale**:

- `10` = decisive positive evidence; `6–9` = positive support;
- `5` = neutral / no opinion;
- `1–4` = **negative feedback** — the validator actively disbelieves
  the pair. `1` is a hard veto.

The final confidence for a candidate is the aggregate over all
validators (e.g. mean, or sum-of-(score−5)). A single `1` from a
hard-veto validator kills the candidate regardless of other scores.
Validators are independent and can be added incrementally.

**Positive signals:**

- **String-set Jaccard** — over rare strings.
- **Shape match** — number of methods, fields, supers, interfaces;
  modifier flags (final/abstract/interface/enum/synthetic).
- **Neighbour overlap** — Jaccard of matched-neighbour sets in both
  directions, per edge type. Inheritance and enclosing-class agreement
  count for much more than method-call agreement.
- **Signature compatibility** — fraction of A's method signatures that
  resolve to a signature present on A' after substituting already-matched
  classes.
- **Bytecode similarity** — mean similarity over the best-matched
  method pairing.
- **Anchor consistency** — referenced framework classes, R-IDs, and
  native symbols agree.

**Hard vetoes (score = 1):**

- Interface ↔ non-interface.
- Enum ↔ non-enum.
- Annotation type ↔ non-annotation.
- Wildly different bytecode size (e.g. > 10× factor) — almost always a
  spurious match from an over-eager string matcher.

**Negative-feedback signals (score 2–4):**

- Big shape delta but not catastrophic — e.g. method count differs by
  > 50% but classes still share strings.
- Inheritance chain disagrees at a level where both supers are already
  matched to different classes.
- Most of A's matched neighbours map to a different class than A'.

**Uniqueness penalty:**

- If the same evidence supports many candidates (e.g. a string that's
  "rare" but actually appears in 30 classes), downweight all candidates
  it supports. Implemented as a per-evidence prior: weight ∝ 1 /
  number-of-candidates-this-evidence-touches.

## Pipeline

```
parse APK A → graph A ─┐
                       ├─→ matcher tier 1 → candidates → validators → confirmed
parse APK B → graph B ─┘                                       │
                                                               ▼
                       ┌── matcher tier 2 ←────── current mapping
                       │
                       ▼
                 candidates → validators → confirmed
                                              │
                                              ▼
                       ┌── matcher tier 3 ←─ updated mapping
                       │           │
                       │           └── iterate to fixpoint
                       ▼
                  final mapping (old → new, with confidence + provenance)
```

**The mapping is NOT monotonic across epochs.** A pair confirmed at
epoch N may be displaced at epoch N+1 if a competing candidate's
confidence exceeds it by at least the engine's displacement margin
(`MutableMapping.displace_margin`, default 0.1). Tier-1 anchors are
locked at insertion (`MutableMapping._locked`) and are immune to
displacement and to validator hard vetoes.

Within a single epoch the mapping is stable — queries are consistent
and the engine guarantees no mid-epoch flips. (An earlier draft of
this spec said "monotonic within a run"; the implementation diverged
intentionally to allow tier-3 propagation to correct earlier
mistakes. The engine is canonical.)

## Data model

- `Class { id, fqn, super, interfaces[], enclosing, modifiers, fields[],
  methods[], strings[], constants[], annotations[], native_methods[] }`
- `Edge { from_class, to_class, kind, site }` where `site` is e.g. the
  method that contains the call, used for fine-grained validators.
- `Candidate { a, b, matcher_id, evidence }` — evidence is a structured
  blob (e.g. `{kind: "unique_string", value: "..."}`) used by validators
  and for human-readable provenance.
- `Mapping = { a_id → (b_id, confidence, [matcher_ids], [evidence]) }`.

## Output

For each class in A, one of:

- A confirmed mapping to a class in B with confidence ≥ threshold,
  annotated with which matchers fired and what evidence each provided.
- A short list of competing candidates with their scores, when no
  candidate dominates.
- Unmatched (likely added/removed between versions).

The provenance is a first-class output, not an afterthought — it's the
only way to debug a wrong mapping or to spot-check confidence.

## Open design questions

1. **Input format — decided: apktool + smali.** Both APKs go through
   `apktool d` and the matcher operates on the smali tree. Reasons:
   text-based smali is trivial to grep/parse for strings, annotations,
   inheritance, and call sites; reuses the existing `smalizator` tool;
   matches the rest of the toolchain in this environment. Bytecode-hash
   matchers normalize smali lines (strip register names, strip labels)
   rather than parsing dex directly.
2. **Member-level matching.** v1 is class-level only. Method/field
   matching falls out naturally once classes are matched (signatures
   become unique within a class once class refs are substituted), so
   it's a follow-on.
3. **Multi-dex handling.** Instagram is multi-dex; class-to-dex
   assignment is mostly stable across builds and could be a weak
   anchor — but only weak, since R8 reshuffles dex buckets when class
   counts change.
4. **Splits.** The 03-16 sample is a split APK (`base.apk` +
   `split_config.xxxhdpi.apk`). Code lives in `base.apk`; resource
   splits don't affect class matching but do affect R-id resolution.
5. **Performance budget.** Instagram is ~50–100k classes per APK. All
   per-pair operations need to be gated by a candidate set — never
   materialize the full A×B cross product. Tier 1 anchors and string
   inverted indexes keep candidate sets small.
6. **Iteration termination.** Tier 3 fixpoint should cap iterations
   (say, 10) and bail out early when an iteration confirms < 0.1% new
   pairs. Beyond that, marginal pairs are noise.

## Test corpus

`samples/ig-2026-02-04/` and `samples/ig-2026-03-16/` — two Instagram
builds ~6 weeks apart. The 03-16 build is unpacked already (`base/`,
`deco/`, `libs/`); the 02-04 build is the raw APK from APKMirror. About
the right diff size: enough churn that naive matching fails, not so much
that ground-truth recovery is impossible.

Ground truth is unobtainable in absolute terms, but we can sanity-check
on:

- Manifest-anchored classes (must round-trip).
- Native-symbol-anchored classes.
- Classes where `@SerializedName` values uniquely pin them.

These give us a few thousand pinned pairs to validate the rest of the
pipeline against.
