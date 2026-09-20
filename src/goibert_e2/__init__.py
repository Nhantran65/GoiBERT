"""Validation-only E2 training ablations.

The E2 package is intentionally separate from :mod:`goibert`.  The latter is
the frozen E1 implementation and is imported here for the data, evaluator,
model-head, optimiser and dataloader contracts.  No E2 entry point loads a
test split.
"""

from .cache import (
    CACHE_SCHEMA,
    TeacherLogitCache,
    build_teacher_logit_cache,
    load_teacher_logit_cache,
)
from .modeling import (
    E2TeacherClassifier,
    build_e2_student,
    build_e2_teacher,
)

_TRAIN_EXPORTS = {
    "E2_MODES",
    "STUDENT_LEARNING_RATES",
    "TEACHER_LEARNING_RATES",
    "train_student_trial",
    "train_teacher_trial",
    "train_trial",
}


def __getattr__(name: str):
    # Keep ``python -m goibert_e2.train`` free of the runpy warning caused by
    # importing the entry module before runpy executes it.
    if name in _TRAIN_EXPORTS:
        from . import train

        return getattr(train, name)
    raise AttributeError(name)

__all__ = [
    "CACHE_SCHEMA",
    "E2TeacherClassifier",
    "E2_MODES",
    "STUDENT_LEARNING_RATES",
    "TEACHER_LEARNING_RATES",
    "TeacherLogitCache",
    "build_e2_student",
    "build_e2_teacher",
    "build_teacher_logit_cache",
    "load_teacher_logit_cache",
    "train_student_trial",
    "train_teacher_trial",
    "train_trial",
]
