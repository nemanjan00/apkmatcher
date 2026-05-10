"""Pure data model — no engine, no I/O, no traversal logic.

Everything in this file is a plain dataclass. Anything that *does* something
(loads a project, walks the graph, drives matchers) lives behind the
interfaces in `apkmatch.interfaces` and is implemented by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


ClassId = str  # opaque per-project identifier (we use the smali FQN)


class EdgeKind(Enum):
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    ENCLOSING = "enclosing"
    FIELD_TYPE = "field_type"
    METHOD_SIG = "method_sig"
    CALL = "call"
    FIELD_ACCESS = "field_access"
    ANNOTATION = "annotation"
    GENERIC_ARG = "generic_arg"
    EXCEPTION = "exception"
    CAST = "cast"
    CLASS_LITERAL = "class_literal"


@dataclass(frozen=True)
class Modifiers:
    public: bool = False
    final: bool = False
    abstract: bool = False
    interface: bool = False
    enum: bool = False
    annotation: bool = False
    synthetic: bool = False


@dataclass(frozen=True)
class Field:
    name: str
    type_ref: str       # e.g. "Lcom/foo/Bar;" or "I"
    modifiers: Modifiers
    static: bool


@dataclass(frozen=True)
class Method:
    name: str
    params: tuple[str, ...]   # type refs, in order
    return_type: str
    modifiers: Modifiers
    native: bool
    # Normalized smali body (registers/labels stripped), or None for abstract/native.
    body_hash: Optional[str] = None


@dataclass(frozen=True)
class Edge:
    """One typed reference from `src` to `dst`.

    `site` is an opaque string identifying *where* in the source class the
    edge originates (e.g. method name, field name) — used by validators that
    care about call-site identity. Engines that don't track sites can leave
    it empty.
    """
    src: ClassId
    dst: ClassId
    kind: EdgeKind
    site: str = ""


@dataclass(frozen=True)
class Class:
    id: ClassId
    fqn: str
    super_class: Optional[ClassId]
    interfaces: tuple[ClassId, ...]
    enclosing: Optional[ClassId]
    modifiers: Modifiers
    fields: tuple[Field, ...]
    methods: tuple[Method, ...]
    strings: tuple[str, ...]            # all string literals appearing in the class
    constants: tuple[int, ...]          # numeric constants worth fingerprinting
    annotations: tuple[str, ...]        # annotation type refs applied to class or members
    native_method_symbols: tuple[str, ...]  # JNI symbol names if any


@dataclass(frozen=True)
class Candidate:
    """A proposed pairing emitted by a matcher.

    Validators consume Candidates without knowing which matcher produced
    them or what loop the engine is running.
    """
    a: ClassId
    b: ClassId
    matcher_id: str
    evidence: tuple = field(default_factory=tuple)


# Validator score scale: 1–10.
#   10 = decisive positive
#   6–9 = positive support
#   5  = neutral / abstain
#   2–4 = negative feedback
#   1  = hard veto (kills the candidate regardless of other validators)
SCORE_VETO = 1
SCORE_NEUTRAL = 5
SCORE_DECISIVE = 10

# Oracle similarity scale: [0.0, 1.0]. NEUTRAL_ANSWER is what the oracle
# returns when it has deferred the real computation to a later epoch
# (because the answer would depend too heavily on yet-unmatched data).
# A neutral answer carries no information — validators should treat it
# the same as score 5 in the 1–10 scale.
NEUTRAL_ANSWER = 0.5
