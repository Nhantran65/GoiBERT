#!/usr/bin/env python3
"""Prepare the standalone eight-intent benchmark without touching the source bundle.

This script is deliberately offline and standard-library only.  The review map below
is an explicit semantic audit of the current customer turn plus preceding customer
turns in the same case.  It never reads or copies assertions, fixtures, primaryIntent,
or expectedIntent into model_input; labels remain target/provenance fields only.

The production classifier prompt is DB-managed and absent from the source snapshot.
The audit therefore follows the checked-in eight-intent taxonomy, the checked-in
prompt excerpt, fast-path order, and classifier tests.  Rows where this evidence is
insufficient are retained conservatively and marked uncertain for root review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/cases.jsonl"
TAXONOMY_SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/src/modules/ai/ai.schema.ts"
PROMPT_SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/src/modules/ai/docs/technicals/prompts_doc/production/INTENT_CLASSIFIER_PROMPT.md"
CONFIG_SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/src/modules/ai/ai.config.ts"
MANIFEST_SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/manifest.json"
POLICY_SOURCE = ROOT / "intent_benchmark/ROOT_LABEL_REVIEW_DECISIONS.md"
LABEL_MAP_SOURCE = ROOT / "intent_benchmark/label_map.json"
OUT = ROOT / "intent_benchmark"

LABELS = {
    "GREETING",
    "CONFIRMATION",
    "GRATITUDE",
    "MENU_QUERY",
    "COMPLAINT",
    "ADDRESS",
    "PHONE",
    "OTHER",
}

SUMMARY = {
    "GREETING": "a greeting or conversational opening",
    "CONFIRMATION": "a clear yes, agreement, approval, or final confirmation",
    "GRATITUDE": "thanks or appreciation",
    "MENU_QUERY": "a concrete menu/product/category/price/availability/recommendation or new product selection",
    "COMPLAINT": "dissatisfaction or a service/problem report",
    "ADDRESS": "delivery-address detail, correction, destination, or delivery-location query",
    "PHONE": "phone-number detail, request, reference, or correction",
    "OTHER": "a request outside the seven intent classes, including payments, invoices, reservations, existing-order status/edits, delivery operations, and neutral cancellation",
}

# Explicit semantic corrections or confirmed conservative decisions.  Keys are stable
# source IDs and zero-based target indices.  Rows absent here were read sequentially;
# their source label is retained when it agrees with the rubric, with a generated
# row-specific rationale and an uncertainty flag where production evidence is weak.
REVIEW_OVERRIDES: dict[tuple[str, int], tuple[str, str, float]] = {
    # Invoice destination is not a delivery address, but the mixed act remains
    # explicitly traceable as a reviewed correction.
    ("pb-v1-0019", 0): ("OTHER", "The address token is supplied as invoice-company information; the taxonomy's ADDRESS class is for delivery-address detail/correction, so invoice handling stays OTHER.", 0.78),
    # Existing-order/reorder examples: the concrete dish/order request keeps the
    # ordering lane even when the case's primaryIntent is an existing-order class.
    ("pb-v1-0028", 0): ("MENU_QUERY", "The latest turn names a concrete dish and requests that product again; under the checked-in FOOD_ORDERING mapping this is a menu/product selection, independent of primaryIntent.", 0.82),
    # Neutral cancellation is operational, while explicit frustration/problems are
    # complaints.  These decisions follow the prompt's dissatisfaction boundary.
    ("pb-v1-0043", 0): ("OTHER", "This is a neutral cancellation request with no dissatisfaction or service problem stated.", 0.96),
    ("pb-v1-0045", 0): ("OTHER", "The customer gives a personal reason for cancelling a reservation and says thanks; no service dissatisfaction is stated.", 0.94),
    ("pb-v1-0047", 0): ("COMPLAINT", "The customer objects that shipping is too expensive before declining the order; that explicit dissatisfaction meets COMPLAINT, but the short mixed act remains uncertain.", 0.78),
    ("pb-v1-0063", 0): ("OTHER", "This asks for the restaurant's address, not a delivery destination or correction; the production prompt's ADDRESS definition is delivery-address-specific, so the location query remains OTHER.", 0.68),
    ("pb-v1-0062", 0): ("OTHER", "This asks for directions to the restaurant, not a delivery destination or correction; the production prompt's ADDRESS definition is delivery-address-specific, so the location query remains OTHER.", 0.68),
    # 0136 remains the original OTHER: it is an existing-order mutation with no
    # standalone product question, and is flagged as a borderline draft edit.
    # Pure updated-price totals stay MENU_QUERY; summaries/recaps and mixed
    # checkout requests stay OTHER.
    ("pb-v1-0214", 4): ("MENU_QUERY", "This is a pure updated-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0215", 4): ("OTHER", "The latest turn asks for an order summary; this is checkout/order state, not menu information.", 0.96),
    ("pb-v1-0216", 4): ("MENU_QUERY", "This is a pure updated-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0217", 4): ("MENU_QUERY", "This is a pure updated-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0218", 4): ("OTHER", "The latest turn asks for a new order summary; this is checkout/order state, not menu information.", 0.96),
    ("pb-v1-0219", 4): ("MENU_QUERY", "This is a pure updated-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0220", 4): ("OTHER", "The latest turn asks for an order summary after a replacement; this is checkout/order state, not menu information.", 0.96),
    ("pb-v1-0221", 4): ("MENU_QUERY", "This is a pure updated-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0222", 2): ("OTHER", "The request to create an order summary is an operational checkout action, not a clear yes/agreement.", 0.9),
    ("pb-v1-0222", 3): ("OTHER", "The request to create an order summary is an operational checkout action, not a clear yes/agreement.", 0.9),
    ("pb-v1-0225", 7): ("MENU_QUERY", "This is a pure total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0226", 7): ("MENU_QUERY", "This is a pure total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0227", 8): ("MENU_QUERY", "This is a pure total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    ("pb-v1-0228", 7): ("OTHER", "The latest turn asks for an order summary; this is checkout/order state, not menu information.", 0.96),
    ("pb-v1-0230", 9): ("MENU_QUERY", "This is a pure final-total query; the checked-in classifier prompt explicitly includes concrete price queries in MENU_QUERY.", 0.9),
    # In an ordering history, closure phrases are clear agreement/confirmation
    # even though they do not authorize a real order on their own.
    ("pb-v1-0181", 3): ("CONFIRMATION", "'đủ rồi' closes the concrete ordering selection in context and is a clear confirmation.", 0.9),
    ("pb-v1-0183", 3): ("CONFIRMATION", "'xong' closes the concrete ordering selection in context and is a clear confirmation.", 0.9),
    ("pb-v1-0184", 3): ("CONFIRMATION", "'được rồi' closes the concrete ordering selection in context and is a clear confirmation.", 0.9),
    ("pb-v1-0185", 3): ("CONFIRMATION", "'vậy thôi' closes the concrete ordering selection in context and is a clear confirmation.", 0.9),
    ("pb-v1-0189", 3): ("CONFIRMATION", "'vậy nha' closes the concrete ordering selection in context and is a clear confirmation.", 0.86),
    # A prior-order cancellation due to an accidental order is operational; it is
    # distinct from a complaint about delay, missing items, or service quality.
    ("pb-v1-0240", 1): ("OTHER", "The customer asks to cancel an earlier order because it was placed accidentally; no service dissatisfaction is stated, while the current order is retained.", 0.84),
    # Mixed greeting/order turn: the action is an order continuation, not a pure
    # opening.  Kept uncertain because classifier prompt bytes are unavailable.
    ("pb-v1-0233", 1): ("GREETING", "The turn opens with a greeting and does not name a concrete product; source GREETING is retained conservatively, with the order-continuation wording flagged.", 0.66),
    ("pb-v1-0229", 7): ("OTHER", "The latest turn requests an order summary; summary/recap requests are operational order state rather than a concrete menu or price query.", 0.86),
}

# Rows where a plausible competing interpretation is material.  Root should review
# these before split finalization; they are not silently used to connect duplicate
# groups.
UNCERTAIN: dict[tuple[str, int], str] = {
    ("pb-v1-0009", 0): "Existing-order add-item request versus menu/product selection; no prior production turn or DB prompt is available, so the conservative source OTHER is retained.",
    ("pb-v1-0037", 0): "Delivery-fee question can be routed as broad business/menu support, but the prompt's dish-and-ordering exclusion suggests OTHER; source label retained pending root decision.",
    ("pb-v1-0038", 0): "Delivery availability/bring-away request is outside dish MENU_QUERY semantically, but source routing may use broad menu support; source label retained pending root decision.",
    ("pb-v1-0039", 0): "Shipping-policy question is outside dish MENU_QUERY semantically, but source routing may use broad menu support; source label retained pending root decision.",
    ("pb-v1-0040", 0): "Delivery-area question is outside dish MENU_QUERY semantically, but source routing may use broad menu support; source label retained pending root decision.",
    ("pb-v1-0041", 0): "Shipping-policy question is outside dish MENU_QUERY semantically, but source routing may use broad menu support; source label retained pending root decision.",
    ("pb-v1-0047", 0): "Shipping-cost dissatisfaction supports COMPLAINT, but the mixed decline/thanks utterance is short and remains a moderate-confidence change.",
    ("pb-v1-0060", 0): "Short 'not gone yet?' can be a neutral status question or frustration; source OTHER retained conservatively.",
    ("pb-v1-0062", 0): "Restaurant directions/location versus delivery-address semantics are separated by the prompt's delivery-specific ADDRESS definition; reviewed OTHER conservatively and flagged for review.",
    ("pb-v1-0063", 0): "Restaurant address versus delivery-address semantics are ambiguous in this anonymized benchmark; reviewed OTHER conservatively under the delivery-specific prompt definition.",
    ("pb-v1-0136", 0): "Existing-order mutation versus concrete product selection is borderline; source OTHER retained because no standalone product query is present.",
    ("pb-v1-0081", 0): "The text may mean 'bring bì cuốn and gỏi upstairs quickly' rather than a lateness complaint; source COMPLAINT retained with uncertainty pending root review.",
    ("pb-v1-0019", 0): "Invoice-company address may resemble an ADDRESS field, but the prompt restricts ADDRESS to delivery details; reviewed OTHER with mixed-act uncertainty.",
    ("pb-v1-0050", 0): "Cancellation wording is noisy and may imply a prior unresolved issue; source COMPLAINT retained until history clarifies whether dissatisfaction is present.",
    ("pb-v1-0139", 0): "Membership-registration link is a service query adjacent to broad menu/ordering support; source MENU_QUERY retained with uncertainty rather than using primaryIntent.",
    ("pb-v1-0172", 2): "Concrete no-onion customization is a draft edit; it could be treated as product selection, but the production prompt is unavailable, so source OTHER is retained conservatively.",
    ("pb-v1-0173", 2): "Cart-content question is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0174", 2): "Quantity restatement is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0175", 1): "Product variant selection ('Original') could be MENU_QUERY, but it is a contextual modifier; source OTHER retained conservatively.",
    ("pb-v1-0175", 2): "Product replacement names a concrete dish but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0176", 1): "Cart-entry question is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0176", 2): "Product replacement names a concrete dish but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0177", 1): "The size question is concrete product information and source MENU_QUERY; context-dependent wording leaves modest uncertainty.",
    ("pb-v1-0177", 2): "Product replacement names a concrete dish but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0178", 2): "Product replacement names a concrete drink but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0179", 2): "Product replacement names a concrete dish but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0179", 3): "Keeping a beverage unchanged can be a draft instruction or a confirmation; source OTHER retained conservatively.",
    ("pb-v1-0180", 2): "Product replacement names a concrete drink but performs an order edit; source OTHER retained conservatively because it is not a standalone menu query.",
    ("pb-v1-0180", 3): "Keeping a food item unchanged can be a draft instruction or a confirmation; source OTHER retained conservatively.",
    ("pb-v1-0181", 1): "Product variant selection could be MENU_QUERY, but it is a contextual clarification after a generic order; source OTHER retained conservatively.",
    ("pb-v1-0182", 1): "Product variant selection could be MENU_QUERY, but it is a contextual clarification after a generic order; source OTHER retained conservatively.",
    ("pb-v1-0183", 1): "Product size selection could be MENU_QUERY, but it is a contextual clarification after a generic order; source OTHER retained conservatively.",
    ("pb-v1-0184", 1): "Product variant selection could be MENU_QUERY, but it is a contextual clarification after a generic order; source OTHER retained conservatively.",
    ("pb-v1-0185", 1): "Product variant selection could be MENU_QUERY, but it is a contextual clarification after a generic order; source OTHER retained conservatively.",
    ("pb-v1-0187", 1): "Removing a concrete item could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0187", 2): "Keeping the remaining items can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0188", 1): "Removing a concrete item could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0188", 2): "Keeping the remaining items can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0189", 1): "Removing a concrete item could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0189", 2): "Cart-content question is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0190", 1): "Removing a concrete item could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0190", 2): "Keeping the remaining items can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0191", 1): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0191", 2): "Quantity correction could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0192", 2): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0193", 1): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0193", 2): "Quantity restatement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0194", 1): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0194", 2): "Keeping a concrete item quantity can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0195", 2): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0195", 3): "Keeping an add-on can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0196", 1): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0196", 2): "Keeping a concrete item can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0197", 1): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0197", 2): "Keeping a concrete item can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0198", 1): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0198", 2): "Keeping a concrete item can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0199", 1): "Concrete customization could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0199", 2): "Keeping a concrete drink unchanged can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0200", 1): "Concrete customization could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0200", 2): "Keeping a concrete drink unchanged can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0201", 1): "Concrete customization could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0201", 2): "Keeping a concrete drink unchanged can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0202", 1): "Concrete customization could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0202", 2): "Keeping a concrete drink unchanged can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0215", 3): "Removing a concrete item could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0216", 3): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0217", 3): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0219", 3): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0220", 3): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0221", 3): "Adding a concrete product could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0223", 2): "Cart-content question is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0226", 3): "Cart-content question is adjacent to MENU_QUERY but may be operational order state; source OTHER retained conservatively.",
    ("pb-v1-0227", 3): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0227", 4): "Keeping a concrete drink quantity can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0228", 3): "Product replacement could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0228", 4): "Keeping a concrete item can be a draft instruction or confirmation; source OTHER retained conservatively.",
    ("pb-v1-0229", 3): "Concrete customization could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0230", 5): "Removing a concrete product could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0230", 6): "Quantity edit could be MENU_QUERY or an operational draft mutation; source OTHER retained conservatively.",
    ("pb-v1-0206", 1): "A district plus shipping-fee question is both delivery-location and policy information; source ADDRESS retained pending root decision.",
    ("pb-v1-0233", 1): "Mixed greeting and order-continuation turn without a named product; source GREETING retained conservatively because the production prompt bytes are unavailable.",
}

MISSING = {
    ("pb-v1-0001", 1),
    ("pb-v1-0004", 1),
    ("pb-v1-0059", 1),
    ("pb-v1-0059", 2),
    ("pb-v1-0061", 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]


def history_for(case: dict[str, Any], turn_index: int) -> list[dict[str, str]]:
    # The source snapshot has no top-level context.  Only prior customer target
    # turns are available; no assistant messages are invented.
    history: list[dict[str, str]] = []
    for turn in case.get("targetTurns", [])[:turn_index]:
        if turn.get("role") != "customer":
            continue
        history.append({"role": "customer", "text": turn.get("text", "")})
    return history


def model_input(case: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turn = case["targetTurns"][turn_index]
    return {
        "messages": history_for(case, turn_index) + [{"role": "customer", "text": turn.get("text", "")}],
    }


def fallback_rationale(label: str, text: str, original: str, history: list[dict[str, str]]) -> str:
    history_note = "preceding customer history was used only to interpret the current turn" if history else "there is no preceding source turn"
    if label == original:
        return f"Independent semantic read: the current turn expresses {SUMMARY[label]}; {history_note}; no stronger competing taxonomy class was found."
    return f"Independent semantic read: the current turn expresses {SUMMARY[label]}, while the source label was {original}; {history_note}; the reviewed label follows the eight-class rubric rather than primaryIntent."


def review_row(case: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turn = case["targetTurns"][turn_index]
    key = (case["caseId"], turn_index)
    original = turn.get("expectedIntent")
    if original not in LABELS:
        raise AssertionError(f"labeled target missing/invalid: {key} -> {original!r}")
    history = history_for(case, turn_index)
    if key in REVIEW_OVERRIDES:
        reviewed, rationale, confidence = REVIEW_OVERRIDES[key]
    else:
        reviewed = original
        rationale = fallback_rationale(reviewed, turn.get("text", ""), original, history)
        confidence = 0.86 if key not in UNCERTAIN else 0.62
    uncertainty = UNCERTAIN.get(key)
    if reviewed not in LABELS:
        raise AssertionError(f"invalid reviewed label: {key} -> {reviewed!r}")
    return {
        "case_id": case["caseId"],
        "turn_index": turn_index,
        "source_ref": case.get("sourceRef"),
        "original_label": original,
        "proposed_label": reviewed,
        "reviewed_label": reviewed,
        "confidence": confidence,
        "uncertain": uncertainty is not None,
        "uncertainty_reason": uncertainty,
        "rationale": rationale,
        "text": turn.get("text", ""),
        "history": history,
    }


def original_row(case: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turn = case["targetTurns"][turn_index]
    return {
        "case_id": case["caseId"],
        "turn_index": turn_index,
        "source_ref": case.get("sourceRef"),
        "role": turn.get("role"),
        "origin": turn.get("origin"),
        "text": turn.get("text", ""),
        "history": history_for(case, turn_index),
        "original_label": turn.get("expectedIntent"),
        "excluded_from_targets": (case["caseId"], turn_index) in MISSING,
    }


def corrected_row(review: dict[str, Any]) -> dict[str, Any]:
    # `model_input` is the only model-facing object.  Source labels and all case
    # contract fields remain outside it, and labels are supervised targets.
    return {
        "sample_id": f"{review['case_id']}:t{review['turn_index']}",
        "case_id": review["case_id"],
        "turn_index": review["turn_index"],
        "source_ref": review["source_ref"],
        "model_input": {
            "messages": review["history"] + [{"role": "customer", "text": review["text"]}],
        },
        "label": review["reviewed_label"],
        "label_provenance": {
            "original_label": review["original_label"],
            "reviewed_label": review["reviewed_label"],
            "confidence": review["confidence"],
            "uncertain": review["uncertain"],
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    cases = read_cases()
    all_targets = [
        (case, i)
        for case in cases
        for i in range(len(case.get("targetTurns", [])))
    ]
    labeled = [(case, i) for case, i in all_targets if (case["caseId"], i) not in MISSING]
    missing = [(case, i) for case, i in all_targets if (case["caseId"], i) in MISSING]
    if len(cases) != 250 or len(all_targets) != 512 or len(labeled) != 507 or len(missing) != 5:
        raise AssertionError((len(cases), len(all_targets), len(labeled), len(missing)))

    originals = [original_row(case, i) for case, i in all_targets]
    reviews = [review_row(case, i) for case, i in labeled]
    corrected = [corrected_row(review) for review in reviews]

    # Stable extraction assertions.
    ids = [f"{r['case_id']}:t{r['turn_index']}" for r in reviews]
    if len(ids) != len(set(ids)) or any(not r["source_ref"] for r in reviews):
        raise AssertionError("unstable or missing source identity")
    for row in corrected:
        for forbidden in ("expectedIntent", "assertions", "fixtures", "primaryIntent", "source_ref"):
            if forbidden in json.dumps(row["model_input"], ensure_ascii=False):
                raise AssertionError(f"input leakage: {row['sample_id']} contains {forbidden}")
        texts = [m["text"] for m in row["model_input"]["messages"]]
        if not texts or texts[-1] != next(r["text"] for r in reviews if f"{r['case_id']}:t{r['turn_index']}" == row["sample_id"]):
            raise AssertionError(f"current message/history malformed: {row['sample_id']}")

    write_jsonl(out / "original_turns.jsonl", originals)
    write_jsonl(out / "review_ledger.jsonl", reviews)
    write_jsonl(out / "corrected_dataset.jsonl", corrected)
    changed = [row for row in reviews if row["original_label"] != row["reviewed_label"]]
    with (out / "review_changes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "case_id", "turn_index", "source_ref", "original_label", "reviewed_label", "confidence", "uncertain", "text", "rationale"],
        )
        writer.writeheader()
        for row in changed:
            writer.writerow({
                "sample_id": f"{row['case_id']}:t{row['turn_index']}",
                "case_id": row["case_id"],
                "turn_index": row["turn_index"],
                "source_ref": row["source_ref"],
                "original_label": row["original_label"],
                "reviewed_label": row["reviewed_label"],
                "confidence": row["confidence"],
                "uncertain": row["uncertain"],
                "text": row["text"],
                "rationale": row["rationale"],
            })
    uncertain = [row for row in reviews if row["uncertain"]]
    with (out / "review_uncertain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "case_id", "turn_index", "source_ref", "text", "history", "original_label", "reviewed_label", "confidence", "uncertainty_reason"],
        )
        writer.writeheader()
        for row in uncertain:
            writer.writerow({
                "sample_id": f"{row['case_id']}:t{row['turn_index']}",
                "case_id": row["case_id"],
                "turn_index": row["turn_index"],
                "source_ref": row["source_ref"],
                "text": row["text"],
                "history": json.dumps(row["history"], ensure_ascii=False, sort_keys=True),
                "original_label": row["original_label"],
                "reviewed_label": row["reviewed_label"],
                "confidence": row["confidence"],
                "uncertainty_reason": row["uncertainty_reason"],
            })

    missing_manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "excluded_count": len(missing),
        "excluded_targets": [
            {
                "sample_id": f"{case['caseId']}:t{i}",
                "case_id": case["caseId"],
                "turn_index": i,
                "source_ref": case.get("sourceRef"),
                "text": case["targetTurns"][i].get("text", ""),
                "reason": "source target turn has no expectedIntent; retained only as history when preceding a labeled target",
            }
            for case, i in missing
        ],
    }
    (out / "missing_exclusion_manifest.json").write_text(json.dumps(missing_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    label_counts = {
        "original": dict(sorted(Counter(r["original_label"] for r in reviews).items())),
        "reviewed": dict(sorted(Counter(r["reviewed_label"] for r in reviews).items())),
        "changed_rows": sum(r["original_label"] != r["reviewed_label"] for r in reviews),
        "uncertain_rows": sum(r["uncertain"] for r in reviews),
    }
    provenance = {
        "source_version": "prompt-bench-v1 bundle snapshot; cases schema prompt-bench-case.v1",
        "bundle_version": json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8")).get("version"),
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "manifest_path": str(MANIFEST_SOURCE.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST_SOURCE),
        "taxonomy_path": str(TAXONOMY_SOURCE.relative_to(ROOT)),
        "taxonomy_sha256": sha256(TAXONOMY_SOURCE),
        "classifier_prompt_path": str(PROMPT_SOURCE.relative_to(ROOT)),
        "classifier_prompt_sha256": sha256(PROMPT_SOURCE),
        "classifier_config_path": str(CONFIG_SOURCE.relative_to(ROOT)),
        "classifier_config_sha256": sha256(CONFIG_SOURCE),
        "review_policy_path": str(POLICY_SOURCE.relative_to(ROOT)),
        "review_policy_sha256": sha256(POLICY_SOURCE),
        "label_map_path": str(LABEL_MAP_SOURCE.relative_to(ROOT)),
        "label_map_sha256": sha256(LABEL_MAP_SOURCE),
        "preparation_script": str(Path(__file__).relative_to(ROOT)),
        "preparation_script_sha256": sha256(Path(__file__)),
        "counts": {"cases": len(cases), "all_target_turns": len(all_targets), "labeled_targets": len(labeled), "missing_targets": len(missing)},
        "labels": label_counts,
        "model_input_contract": {
            "current_turn_only_target": True,
            "history_roles_preserved": ["customer"],
            "assistant_messages_invented": False,
            "forbidden_fields": ["expectedIntent", "assertions", "fixtures", "primaryIntent", "source_ref"],
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Audit-friendly counts for root before grouped split finalization.
    cross = {label: Counter() for label in LABELS}
    for r in reviews:
        cross[r["original_label"]][r["reviewed_label"]] += 1
    (out / "label_crosstab.json").write_text(json.dumps({k: dict(sorted(v.items())) for k, v in sorted(cross.items())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Current-text and full-input duplicate evidence.  Generic current-text overlap
    # is reported separately and never becomes a grouping edge by itself.
    current_groups: dict[str, list[str]] = defaultdict(list)
    input_groups: dict[str, list[str]] = defaultdict(list)
    for row in corrected:
        sample = row["sample_id"]
        current = normalize(row["model_input"]["messages"][-1]["text"])
        full = normalize("\n".join(f"{m['role']}:{m['text']}" for m in row["model_input"]["messages"]))
        current_groups[current].append(sample)
        input_groups[full].append(sample)
    generic_current = []
    for key, samples in sorted(current_groups.items()):
        if len(samples) > 1:
            generic_current.append({"normalized_current_text": key, "samples": samples, "count": len(samples), "grouping_policy": "report_only_unless_full_input_or_substantive_family_confirmed"})
    exact_full = [{"normalized_full_input": key, "samples": samples, "count": len(samples), "confirmed_group": True} for key, samples in sorted(input_groups.items()) if len(samples) > 1]
    duplicate_report = {
        "normalization": "NFC then casefold then collapse whitespace",
        "current_text_overlap_families": generic_current,
        "exact_full_input_duplicate_families": exact_full,
        "near_duplicate_candidates": [],
        "policy": "generic acknowledgments and repeated placeholder-only address turns are reported separately; only exact full-input duplicates or confirmed substantive conversation families may join splits",
    }
    (out / "duplicate_analysis.json").write_text(json.dumps(duplicate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"cases": len(cases), "all_targets": len(all_targets), "labeled": len(labeled), "missing": len(missing), **label_counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
