"""apkmatch — match obfuscated classes across APK versions.

Engine-independent core. See:
  * model.py       — pure data
  * interfaces.py  — Matcher, CandidateValidator, NeighbourRelation, views
  * (engine.py to come — in-memory implementation)
"""

from .model import (
    Candidate,
    Class,
    ClassId,
    Edge,
    EdgeKind,
    Field,
    Method,
    Modifiers,
    NEUTRAL_ANSWER,
    SCORE_DECISIVE,
    SCORE_NEUTRAL,
    SCORE_VETO,
)
from .interfaces import EpochChurn
from .interfaces import (
    CandidateValidator,
    MappingView,
    Matcher,
    NeighbourRelation,
    Oracle,
    ProjectView,
    ResolveRef,
    ScoreAggregator,
)

__all__ = [
    "Candidate", "Class", "ClassId", "Edge", "EdgeKind",
    "Field", "Method", "Modifiers",
    "NEUTRAL_ANSWER", "SCORE_DECISIVE", "SCORE_NEUTRAL", "SCORE_VETO",
    "CandidateValidator", "EpochChurn", "MappingView", "Matcher",
    "NeighbourRelation", "Oracle", "ProjectView", "ResolveRef", "ScoreAggregator",
]
