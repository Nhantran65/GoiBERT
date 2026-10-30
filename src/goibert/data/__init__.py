"""Benchmark loaders.

The eight tasks fall into two loader families:

* :mod:`goibert.data.uit`: ``vsfc_sentiment``, ``vsfc_topic``, ``visfd``,
  ``absa_restaurant``, ``absa_hotel``;
* :mod:`goibert.data.extra`: ``vispam``, ``vinli``, ``phoner``.
"""

from . import extra, uit

UIT_TASKS = uit.TASK_IDS
EXTRA_TASKS = extra.TASK_IDS
TASKS = UIT_TASKS + EXTRA_TASKS


def family(task_id: str) -> str:
    if task_id in UIT_TASKS:
        return "uit"
    if task_id in EXTRA_TASKS:
        return "extra"
    raise ValueError(f"Unknown task {task_id!r}; choose from {list(TASKS)!r}")
