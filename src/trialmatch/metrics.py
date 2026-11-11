from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, Any]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted label sequences must have the same length")
    if not gold_labels:
        raise ValueError("Cannot calculate classification metrics for an empty split")
    all_labels = [*gold_labels, *predicted_labels]
    if any(not isinstance(label, str) or not label.strip() for label in all_labels):
        raise ValueError("Classification labels must be non-empty strings")

    classes = sorted(set(gold_labels) | set(predicted_labels))
    confusion = {gold: {predicted: 0 for predicted in classes} for gold in classes}
    for gold, predicted in zip(gold_labels, predicted_labels):
        confusion[gold][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in classes:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in classes if other != label)
        false_negative = sum(confusion[label][other] for other in classes if other != label)
        scores = _prf(true_positive, false_positive, false_negative)
        per_class[label] = {
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "support": sum(confusion[label].values()),
        }

    accuracy = sum(
        gold == predicted for gold, predicted in zip(gold_labels, predicted_labels)
    ) / len(gold_labels)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(scores["f1"] for scores in per_class.values()) / len(classes),
        "labels": classes,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "count": len(gold_labels),
    }


def evidence_metrics(
    gold_evidence: Sequence[Iterable[int]],
    predicted_evidence: Sequence[Iterable[int]],
) -> dict[str, Any]:
    if len(gold_evidence) != len(predicted_evidence):
        raise ValueError("Gold and predicted evidence sequences must have the same length")
    if not gold_evidence:
        raise ValueError("Cannot calculate evidence metrics for an empty split")

    true_positive = false_positive = false_negative = 0
    record_scores: list[dict[str, float]] = []
    empty_gold_records = empty_gold_exact_matches = 0
    empty_gold_with_predictions = 0
    for gold_values, predicted_values in zip(gold_evidence, predicted_evidence):
        gold = set(gold_values)
        predicted = set(predicted_values)
        if any(type(sentence_id) is not int or sentence_id < 0 for sentence_id in gold | predicted):
            raise ValueError("Evidence IDs must be non-negative integers")
        matched = gold & predicted
        record_tp = len(matched)
        record_fp = len(predicted - gold)
        record_fn = len(gold - predicted)
        true_positive += record_tp
        false_positive += record_fp
        false_negative += record_fn

        if not gold:
            empty_gold_records += 1
            if not predicted:
                empty_gold_exact_matches += 1
                record_scores.append({"precision": 1.0, "recall": 1.0, "f1": 1.0})
            else:
                empty_gold_with_predictions += 1
                record_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        else:
            record_scores.append(_prf(record_tp, record_fp, record_fn))

    micro = _prf(true_positive, false_positive, false_negative)
    if true_positive + false_positive + false_negative == 0:
        micro = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {
        "micro": micro,
        "macro": {
            metric: sum(scores[metric] for scores in record_scores) / len(record_scores)
            for metric in ("precision", "recall", "f1")
        },
        "empty_gold_records": empty_gold_records,
        "empty_gold_exact_matches": empty_gold_exact_matches,
        "empty_gold_with_predictions": empty_gold_with_predictions,
        "count": len(gold_evidence),
    }
