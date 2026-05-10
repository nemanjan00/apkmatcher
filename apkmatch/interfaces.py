"""Engine-independent interfaces.

The contract here is:

  * Matchers and validators NEVER touch the filesystem, NEVER walk a
    directory, NEVER decide when they get called. They consume the views
    defined below and emit Candidates / scores.

  * NeighbourRelations are pure functions from a Class (+ a lookup) to a
    set of edges. They don't know what graph algorithm uses them.

  * The engine — whatever shape it takes (single-pass, fixpoint loop,
    parallel workers, distributed) — is the only thing that knows how to
    load a project, store a mapping, and orchestrate the pieces. It
    implements ProjectView and MappingView and hands them to matchers and
    validators.

This separation means a matcher written today still works if we later
swap apktool/smali for direct dex parsing, or if we switch from a
fixpoint loop to a constraint solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Protocol, runtime_checkable

from .model import Candidate, Class, ClassId, Edge, EdgeKind, Method


# --------------------------------------------------------------------------- #
# Views — the engine implements these; everyone else consumes them.
# --------------------------------------------------------------------------- #

@runtime_checkable
class ProjectView(Protocol):
    """Read-only view of one parsed APK.

    Matchers and validators call into this to inspect classes and edges.
    They MUST NOT assume anything about how the underlying data is stored
    (in memory, on disk, in a database, lazily reparsed from smali, …).
    """

    def classes(self) -> Iterator[Class]: ...

    def get(self, cid: ClassId) -> Optional[Class]: ...

    def neighbours(
        self, cid: ClassId, kind: Optional[EdgeKind] = None
    ) -> Iterator[Edge]:
        """Outgoing edges from `cid`, optionally filtered by kind."""
        ...

    def reverse_neighbours(
        self, cid: ClassId, kind: Optional[EdgeKind] = None
    ) -> Iterator[Edge]:
        """Incoming edges to `cid`."""
        ...

    # Indexed lookups that matchers rely on. Implementations are free to
    # build these eagerly or lazily — that's an engine concern.

    def classes_containing_string(self, s: str) -> Iterator[ClassId]: ...

    def classes_with_native_symbol(self, sym: str) -> Iterator[ClassId]: ...

    def classes_with_annotation(self, ann_ref: str) -> Iterator[ClassId]: ...

    def string_frequency(self, s: str) -> int:
        """How many classes in this project contain the literal `s`."""
        ...


@runtime_checkable
class MappingView(Protocol):
    """Read-only view of the current A→B mapping.

    The mapping is **not monotonic across epochs**. A pair confirmed at
    epoch N may be revoked at epoch N+1 if better evidence emerges (a
    competing candidate with confidence higher by at least the engine's
    displacement margin). Validators must therefore re-score candidates
    they care about every epoch, and the oracle invalidates caches
    keyed on `epoch`.

    Within a single epoch the mapping is stable — queries are
    consistent and the engine guarantees no mid-epoch flips.
    """

    @property
    def epoch(self) -> int:
        """Monotonically increasing counter. Bumps when the engine
        commits a batch of mapping changes (confirmations, revocations,
        confidence rescores). Use as a cache key."""
        ...

    def get(self, a: ClassId) -> Optional[ClassId]: ...

    def inverse(self, b: ClassId) -> Optional[ClassId]: ...

    def confidence(self, a: ClassId) -> Optional[float]: ...

    def is_confirmed(self, a: ClassId, b: ClassId) -> bool:
        """A pair is 'confirmed' iff confidence ≥ the engine's
        confirmation threshold AT THIS EPOCH. May become unconfirmed
        next epoch."""
        ...

    def is_locked(self, a: ClassId) -> bool:
        """A locked mapping cannot be revoked even by a higher-confidence
        candidate. Engines lock pairs once they've survived several
        epochs unchanged at near-max confidence (or when an anchor
        matcher with a hard guarantee — e.g. JNI symbol — fired)."""
        ...

    def __iter__(self) -> Iterator[tuple[ClassId, ClassId]]: ...

    # ---- Convergence signals --------------------------------------------

    @property
    def churn(self) -> "EpochChurn":
        """How much the mapping changed in the most recent epoch
        transition. The engine uses this to decide when to stop
        iterating; matchers/validators may also read it to back off
        expensive work near convergence."""
        ...


@dataclass(frozen=True)
class EpochChurn:
    """Diff between two consecutive epochs of the mapping.

    A run has **converged** when several consecutive epochs report
    churn satisfying the engine's convergence criterion — typically:

      * `confirmed_added + confirmed_removed + confirmed_changed
         <= max_changes`, AND
      * `mean_confidence_delta <= max_drift`, AND
      * `pending_deferrals` is 0 or has not decreased,

    for K epochs in a row (K ≥ 2 to avoid declaring victory on a
    single quiet pass through a still-evolving mapping).

    The `pending_deferrals` clause matters: an epoch may look quiet on
    confirmations alone while the oracle still has open questions
    waiting on data that *will* arrive. The engine should keep
    iterating until the deferral queue stops shrinking — at which
    point unanswered deferrals are answered with NEUTRAL_ANSWER
    permanently and the run terminates.
    """
    epoch: int
    confirmed_added: int       # pairs newly above the confirmation threshold
    confirmed_removed: int     # pairs that fell below or were displaced
    confirmed_changed: int     # confirmed A whose B target moved
    candidates_emitted: int    # total candidates this epoch (for diagnostics)
    mean_confidence_delta: float  # mean |Δconfidence| over confirmed pairs
    pending_deferrals: int     # oracle questions waiting for a future epoch
    deferrals_resolved: int    # questions that got real answers this epoch


# --------------------------------------------------------------------------- #
# Oracle — derived-measurement queries usable by both matchers and validators.
# --------------------------------------------------------------------------- #

@runtime_checkable
class Oracle(Protocol):
    """Asks higher-level questions about the two projects + current mapping.

    Validators and matchers use the Oracle instead of recomputing the same
    similarity metrics from raw data over and over. The engine owns the
    implementation, which means it can:

      * cache repeated queries within an iteration;
      * memoize across iterations and invalidate on mapping change;
      * break cycles when one query recursively triggers another;
      * upgrade a cheap-but-rough estimate to a precise one as the
        mapping fills in, without any caller knowing.

    All methods return values in [0.0, 1.0] unless documented otherwise.
    A return of 1.0 means "as similar as possible given current evidence."
    Callers MUST treat the oracle as advisory — it can return a stale
    cached value mid-iteration. The engine guarantees consistency only at
    iteration boundaries.

    --- Deferred answers ---

    When a question's honest answer would depend heavily on yet-unmatched
    data, the oracle returns `NEUTRAL_ANSWER` (0.5) and queues the
    question for the next epoch. The engine tracks which validator/matcher
    invocation triggered the deferred question and re-runs that
    invocation when the question's inputs change.

    --- Deadlock-avoidance contract (engine MUST guarantee) ---

    The deferral mechanism can deadlock if it isn't bounded. The engine
    is required to enforce, in this order:

      1. **Bounded recursion.** A single oracle call has a fixed maximum
         recursion depth. Beyond it, the answer collapses to a cheap
         non-recursive baseline (e.g. raw shape similarity). No oracle
         method may call itself transitively without the depth counter
         being decremented.

      2. **Cycle break on hypothetical queries.** A `hypothetical_*`
         query for (a, b) that, while resolving, asks again about
         (a, b) — directly or via neighbours — returns the cheap
         baseline immediately for the inner call.

      3. **Per-question defer budget.** Each distinct deferred question
         has a max-defer count (default 8). After that, it is answered
         with NEUTRAL_ANSWER permanently and removed from the queue,
         even if its dependencies are still moving.

      4. **Forward-progress invariant.** Every epoch must either
         (a) confirm/revoke at least one pair, OR (b) decrease
         `pending_deferrals`. If an epoch does neither, the engine
         force-resolves all remaining deferrals to NEUTRAL_ANSWER and
         terminates the run. This is the ultimate deadlock breaker —
         it cannot silently spin forever.

      5. **Stable scheduling.** Deferrals are drained in FIFO order.
         The engine MUST NOT reorder them based on validator priority
         mid-run; this prevents starvation cycles where two questions
         each wait for the other.
    """

    def defer(self, reason: str = "") -> float:
        """Explicit deferral. Returns NEUTRAL_ANSWER and registers a
        pending question against the current scoring context (the
        engine sets that context when it invokes a validator/matcher).

        Validators rarely call this directly — they call the typed
        methods below and let the oracle decide whether to defer
        internally. Direct use is for custom validators that have
        their own notion of 'I can't answer this yet'."""
        ...

    def has_pending(self) -> bool:
        """True if the deferral queue is non-empty. Used by the engine
        to drive convergence; matchers can use it to suppress expensive
        speculative work near the end of a run."""
        ...

    # ---- Class-level questions -------------------------------------------

    def class_similarity(self, a: ClassId, b: ClassId) -> float:
        """Overall best-current-estimate similarity of two classes.
        Aggregates whatever the engine deems cheap (string overlap, shape,
        neighbour overlap). Validators that want a single 'how close are
        these?' number should call this rather than rolling their own."""
        ...

    def shape_similarity(self, a: ClassId, b: ClassId) -> float:
        """Modifier flags + counts of methods/fields/supers/interfaces."""
        ...

    def string_similarity(self, a: ClassId, b: ClassId) -> float:
        """TF-IDF-weighted Jaccard over string literals (rare strings
        weigh more). The engine owns the IDF table."""
        ...

    # ---- Neighbour-level questions ---------------------------------------

    def neighbour_overlap(
        self,
        a: ClassId,
        b: ClassId,
        kind: Optional[EdgeKind] = None,
    ) -> float:
        """How well do A's outgoing neighbours line up with B's, under
        the current mapping? Already-matched neighbours count fully;
        unmatched-but-mutually-similar neighbours count fractionally
        (the engine decides the weighting). Filter by edge kind, or
        pass None for a weighted blend across kinds."""
        ...

    def reverse_neighbour_overlap(
        self,
        a: ClassId,
        b: ClassId,
        kind: Optional[EdgeKind] = None,
    ) -> float:
        """Same, on incoming edges (who references this class)."""
        ...

    # ---- Method-level questions ------------------------------------------

    def method_similarity(
        self,
        a_class: ClassId, a_method: Method,
        b_class: ClassId, b_method: Method,
    ) -> float:
        """Bytecode-hash equality, signature shape after substituting
        already-matched class refs, and (optionally) fuzzy bytecode diff."""
        ...

    def best_method_pairing(
        self, a: ClassId, b: ClassId,
    ) -> float:
        """Mean similarity of A's methods to their best match in B
        (Hungarian-style, but engine may approximate)."""
        ...

    # ---- Speculative / what-if -------------------------------------------

    def hypothetical_class_similarity(
        self, a: ClassId, b: ClassId,
    ) -> float:
        """Like `class_similarity`, but answers 'how would this score if
        we tentatively assumed (a, b) were matched?' Used by tier-3
        propagation matchers and by lock-step reasoning. The engine is
        responsible for not blowing up on cycles (a→b→a queries collapse
        to the cheap baseline)."""
        ...


# --------------------------------------------------------------------------- #
# Concept 1 — neighbour relations
# --------------------------------------------------------------------------- #

class NeighbourRelation(Protocol):
    """Extracts edges of a single kind from a class.

    Engine-independent: it gets a Class plus a `resolve` callable for
    looking up other classes by smali type ref, and yields Edges. It does
    not know whether the engine will store the edges in a graph, stream
    them, or recompute on demand.
    """

    kind: EdgeKind
    weight: float  # how much to trust this edge type when validating

    def edges(
        self,
        cls: Class,
        resolve: "ResolveRef",
    ) -> Iterable[Edge]: ...


class ResolveRef(Protocol):
    """Maps a smali type ref (`Lcom/foo/Bar;`) to a ClassId in the same
    project, or None if it's external (framework, library)."""

    def __call__(self, type_ref: str) -> Optional[ClassId]: ...


# --------------------------------------------------------------------------- #
# Concept 2 — matchers
# --------------------------------------------------------------------------- #

class Matcher(Protocol):
    """Proposes candidate (A→B) pairings.

    A matcher is given both project views and the current mapping and
    yields Candidates. It does NOT:

      * decide when it runs (the engine schedules tiers and iterations);
      * confirm pairs (only validators + the engine combine into a
        confidence and decide on confirmation);
      * mutate the mapping (the engine owns mutation).
    """

    id: str
    tier: int  # 1 = anchors, 2 = content fingerprints, 3 = structural

    def propose(
        self,
        a: ProjectView,
        b: ProjectView,
        mapping: MappingView,
        oracle: Oracle,
    ) -> Iterator[Candidate]: ...


# --------------------------------------------------------------------------- #
# Concept 3 — candidate validators
# --------------------------------------------------------------------------- #

class CandidateValidator(Protocol):
    """Rates a single candidate on the 1–10 scale.

    A validator does NOT know:

      * which matcher produced the candidate (only `candidate.matcher_id`
        is visible, for logging — not for branching);
      * whether the engine is running tier 2 or tier 3;
      * whether other validators will agree;
      * what "confirmation threshold" the engine will apply.

    Score semantics (see model.py):
      10 decisive positive · 6–9 positive · 5 neutral · 2–4 negative · 1 veto.
    """

    id: str

    def score(
        self,
        candidate: Candidate,
        a: ProjectView,
        b: ProjectView,
        mapping: MappingView,
        oracle: Oracle,
    ) -> int: ...


# --------------------------------------------------------------------------- #
# Score aggregation — also engine-independent.
# --------------------------------------------------------------------------- #

class ScoreAggregator(Protocol):
    """Combines per-validator scores into a final confidence.

    Pulled out as its own interface so the engine can swap aggregation
    strategies (mean, weighted sum, log-product, learned model) without
    changing any validator.
    """

    def aggregate(self, scores: list[tuple[str, int]]) -> float:
        """Returns confidence in [0, 1]. A score of 1 from any validator
        is a hard veto and MUST yield 0.0."""
        ...
