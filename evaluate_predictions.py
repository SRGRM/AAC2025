"""Score saved API outputs against ground-truth labels with the challenge metrics.

Detection::

    python evaluate_predictions.py detection --predictions out/detection_track.json --labels labels.txt

``labels.txt`` has one image per line: ``<image name> <label> [attack]``, where the label
is 1 / adversarial / attack or 0 / clean / bona_fide. The script reports APCER, BPCER,
F1 and ROC AUC (the challenge metrics) and the EER, plus the detection rate per attack
when the third column is present. Decisions are taken from the predictions unless
``--threshold`` is given.

Resilience::

    python evaluate_predictions.py resilience --predictions out/resilience_track.json --labels pairs.txt

``pairs.txt`` has one pair per line: ``<image 1> <image 2> <label> [attacked] [scenario]``,
where the label is 1 / genuine or 0 / impostor, ``attacked`` is 1 when one image of the
pair was attacked (default 0) and ``scenario`` is a free name such as ``evasion_pifgsm``.
The script reports the EER over all pairs, the bona fide EER, the attack success rate
(ASR) on the attacked pairs at the operating threshold and the robustness score
(1 - EER) * (1 - ASR), overall and per scenario. The operating threshold is
``--threshold`` when given, otherwise the EER threshold of the bona fide pairs.

Label entries are matched to the predicted image paths by their trailing path components,
so label files can use paths relative to images_dir (e.g. ``pair_12/im_0.png``); an entry
that fits several predicted images is reported as an error.

The challenge paper names the metrics but not every detail of their computation. Here the
EER is computed over all pairs of the label file and the ASR at the bona fide EER
threshold; these choices are this script's, not the organisers', so results on the
official test set may differ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import f1_score

from aac.checkpoint import write_json
from aac.metrics import attack_success_rate, equal_error_rate, robustness_score, roc_auc

LABELS = {
    "1": 1,
    "adversarial": 1,
    "attack": 1,
    "genuine": 1,
    "0": 0,
    "clean": 0,
    "bona_fide": 0,
    "bonafide": 0,
    "impostor": 0,
    "imposter": 0,
}


def read_rows(path: str, min_fields: int) -> List[Tuple[int, List[str]]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.replace(",", " ").split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < min_fields:
                raise ValueError(f"{path}, line {number}: expected at least {min_fields} fields")
            rows.append((number, fields))
    return rows


def parse_label(token: str, path: str, number: int) -> int:
    try:
        return LABELS[token.strip().lower()]
    except KeyError:
        raise ValueError(f"{path}, line {number}: unknown label {token!r}") from None


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        predictions = json.load(handle)
    if not isinstance(predictions, list):
        raise ValueError(f"{path} should contain a JSON list")
    return predictions


PathKey = Tuple[str, ...]


def path_key(path: str) -> PathKey:
    """Path components, independent of separators and leading './'."""
    return tuple(part for part in str(path).replace("\\", "/").split("/") if part not in ("", "."))


class PathResolver:
    """Finds the predicted image whose path ends with a path given in a label file."""

    def __init__(self, keys: Iterable[PathKey]) -> None:
        self.by_suffix: Dict[PathKey, Set[PathKey]] = defaultdict(set)
        for key in keys:
            for start in range(len(key)):
                self.by_suffix[key[start:]].add(key)

    def resolve(self, path: str) -> Optional[PathKey]:
        matches = self.by_suffix.get(path_key(path), set())
        if len(matches) > 1:
            raise ValueError(
                f"{path!r} matches {len(matches)} predicted images; "
                "use paths relative to images_dir in the label file"
            )
        return next(iter(matches)) if matches else None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def print_table(rows: Sequence[Tuple[str, ...]]) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100 * value:.2f}"


def evaluate_detection(args: argparse.Namespace) -> Dict[str, Any]:
    """APCER, BPCER, F1, AUC and EER of saved detection outputs."""
    by_path = {path_key(p.get("image_path", "")): p for p in load_predictions(args.predictions)}
    resolver = PathResolver(by_path)
    labels, scores, decisions, attacks = [], [], [], []
    missing, failed = 0, 0
    for number, fields in read_rows(args.labels, 2):
        key = resolver.resolve(fields[0])
        prediction = None if key is None else by_path[key]
        if prediction is None:
            missing += 1
            continue
        if not is_number(prediction.get("score")):
            failed += 1
            continue
        labels.append(parse_label(fields[1], args.labels, number))
        scores.append(float(prediction["score"]))
        decisions.append(prediction.get("decision"))
        attacks.append(fields[2] if len(fields) > 2 else None)
    if not labels:
        raise ValueError("No labelled image has a valid prediction")

    labels_array, scores_array = np.asarray(labels), np.asarray(scores)
    if args.threshold is not None:
        flagged = scores_array >= args.threshold
    elif all(isinstance(decision, bool) for decision in decisions):
        flagged = np.asarray(decisions)
    else:
        raise ValueError("Some predictions have no boolean decision; pass --threshold")

    attacked, bona_fide = labels_array == 1, labels_array == 0
    eer, _ = equal_error_rate(labels_array, scores_array)
    metrics = {
        "images": int(labels_array.size),
        "missing_predictions": missing,
        "failed_predictions": failed,
        "apcer": float(np.mean(~flagged[attacked])) if attacked.any() else float("nan"),
        "bpcer": float(np.mean(flagged[bona_fide])) if bona_fide.any() else float("nan"),
        "f1": float(f1_score(labels_array, flagged.astype(int), zero_division=0)),
        "auc": roc_auc(labels_array, scores_array),
        "eer": eer,
    }
    per_attack: Dict[str, List[bool]] = defaultdict(list)
    for attack, is_flagged, label in zip(attacks, flagged, labels):
        if attack is not None and label == 1:
            per_attack[attack].append(bool(is_flagged))
    metrics["detection_rate_per_attack"] = {
        k: float(np.mean(v)) for k, v in sorted(per_attack.items())
    }

    print(f"Detection: {metrics['images']} images ({missing} without prediction, {failed} failed)")
    print_table(
        [
            ("metric", "value"),
            ("APCER", f"{metrics['apcer']:.4f}"),
            ("BPCER", f"{metrics['bpcer']:.4f}"),
            ("F1", f"{metrics['f1']:.4f}"),
            ("AUC-ROC", f"{metrics['auc']:.4f}"),
            ("EER", f"{metrics['eer']:.4f}"),
        ]
    )
    if per_attack:
        print("\nDetection rate per attack")
        print_table(
            [("attack", "rate")]
            + [(k, f"{v:.4f}") for k, v in metrics["detection_rate_per_attack"].items()]
        )
    return metrics


def evaluate_resilience(args: argparse.Namespace) -> Dict[str, Any]:
    """EER, ASR and robustness score of saved resilience outputs, overall and per scenario."""
    by_pair: Dict[Tuple[PathKey, PathKey], Dict[str, Any]] = {}
    for prediction in load_predictions(args.predictions):
        first = path_key(prediction.get("image_path_1", ""))
        second = path_key(prediction.get("image_path_2", ""))
        by_pair[(first, second)] = prediction
        by_pair.setdefault((second, first), prediction)
    resolver = PathResolver({key for pair in by_pair for key in pair})

    labels, scores, attacked, scenarios = [], [], [], []
    missing, failed = 0, 0
    for number, fields in read_rows(args.labels, 3):
        first, second = resolver.resolve(fields[0]), resolver.resolve(fields[1])
        prediction = by_pair.get((first, second)) if first and second else None
        if prediction is None:
            missing += 1
            continue
        if not is_number(prediction.get("similarity")):
            failed += 1
            continue
        labels.append(parse_label(fields[2], args.labels, number))
        scores.append(float(prediction["similarity"]))
        attacked.append(len(fields) > 3 and fields[3] not in ("0", "false", "no"))
        scenarios.append(fields[4] if len(fields) > 4 else ("attacked" if attacked[-1] else None))
    if not labels:
        raise ValueError("No labelled pair has a valid prediction")

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    attacked_array = np.asarray(attacked, dtype=bool)
    bona_fide = ~attacked_array

    eer_all, threshold_all = equal_error_rate(labels_array, scores_array)
    eer_bona_fide, threshold_bona_fide = equal_error_rate(
        labels_array[bona_fide], scores_array[bona_fide]
    )
    if args.threshold is not None:
        threshold, source = float(args.threshold), "--threshold"
    elif math.isfinite(threshold_bona_fide):
        threshold, source = threshold_bona_fide, "bona fide EER"
    else:
        threshold, source = threshold_all, "EER over all pairs"
    asr = attack_success_rate(labels_array[attacked_array], scores_array[attacked_array], threshold)

    metrics: Dict[str, Any] = {
        "pairs": int(labels_array.size),
        "attacked_pairs": int(attacked_array.sum()),
        "missing_predictions": missing,
        "failed_predictions": failed,
        "threshold": threshold,
        "threshold_source": source,
        "eer": eer_all,
        "eer_bona_fide": eer_bona_fide,
        "asr": asr,
        "robustness_score": robustness_score(eer_all, asr),
        "scenarios": {},
    }
    table = [("scenario", "pairs", "ASR %", "EER %", "RS %")]
    for scenario in sorted({s for s in scenarios if s is not None}):
        chosen = np.asarray([s == scenario for s in scenarios]) & attacked_array
        subset = chosen | bona_fide
        scenario_eer, _ = equal_error_rate(labels_array[subset], scores_array[subset])
        scenario_asr = attack_success_rate(labels_array[chosen], scores_array[chosen], threshold)
        metrics["scenarios"][scenario] = {
            "pairs": int(chosen.sum()),
            "asr": scenario_asr,
            "eer": scenario_eer,
            "robustness_score": robustness_score(scenario_eer, scenario_asr),
        }
        table.append(
            (
                scenario,
                str(int(chosen.sum())),
                percent(scenario_asr),
                percent(scenario_eer),
                percent(robustness_score(scenario_eer, scenario_asr)),
            )
        )

    print(
        f"Resilience: {metrics['pairs']} pairs, {metrics['attacked_pairs']} attacked "
        f"({missing} without prediction, {failed} failed)"
    )
    print(f"Operating threshold {threshold:.4f} ({source})")
    print_table(
        [
            ("metric", "value"),
            ("EER (all pairs)", percent(eer_all) + " %"),
            ("EER (bona fide)", percent(eer_bona_fide) + " %"),
            ("ASR", percent(asr) + " %"),
            ("Robustness score", f"{metrics['robustness_score']:.4f}"),
        ]
    )
    if len(table) > 1:
        print("\nPer scenario (EER over the scenario's attacked pairs plus all bona fide pairs)")
        print_table(table)
    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions with the challenge metrics"
    )
    parser.add_argument("track", choices=("detection", "resilience"))
    parser.add_argument(
        "--predictions", required=True, help="detection_track.json or resilience_track.json"
    )
    parser.add_argument(
        "--labels", required=True, help="ground-truth label file (see the module docstring)"
    )
    parser.add_argument("--threshold", type=float, default=None, help="operating threshold to use")
    parser.add_argument("--output", default=None, help="optional JSON file for the metrics")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    evaluate = evaluate_detection if args.track == "detection" else evaluate_resilience
    try:
        metrics = evaluate(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output:
        write_json(metrics, args.output)
        print(f"\nMetrics written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
