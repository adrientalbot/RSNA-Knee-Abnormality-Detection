"""Generate tri-state RSNA knee labels from reports with TypeSafe JEV."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

TARGET_DESCRIPTIONS = {
    "ACL": "an abnormal anterior cruciate ligament, including tear, sprain, or degeneration",
    "MCL": "an abnormal medial collateral ligament, including tear or sprain",
    "Medial Meniscus": "an abnormal medial meniscus, including tear, degeneration, or prior resection",
    "Lateral Meniscus": "an abnormal lateral meniscus, including tear, degeneration, or prior resection",
    "Medial OA": "medial tibiofemoral osteoarthritis or medial-compartment degenerative cartilage disease",
    "Lateral OA": "lateral tibiofemoral osteoarthritis or lateral-compartment degenerative cartilage disease",
    "PF OA": "patellofemoral osteoarthritis or patellofemoral degenerative cartilage disease",
    "Effusion": "knee joint effusion or abnormal intra-articular fluid",
    "Synovitis": "synovitis or abnormal synovial thickening or proliferation",
    "Baker's": "a Baker or popliteal cyst",
    "Contusion": "bone contusion, bone bruise, or traumatic bone-marrow edema",
    "Fracture": "an acute, stress, insufficiency, avulsion, or subchondral fracture",
}

CHECKPOINT_VERSION = 2
VALID_VERDICTS = {"YES", "NO", "UNK"}
# `api_labeler.py` maps NO and UNK to 0.08 and 0.28. Its positive score also
# uses severity (0.68/0.82/0.94), which the completed JEV Choice run did not
# request, so an unqualified/moderate YES uses the middle positive anchor.
SCORE_ANCHORS = {"NO": 0.08, "UNK": 0.28, "YES": 0.82}
RETRY_POLICY = RetryPolicy(
    max_retries=4,
    backoff_initial=1.0,
    backoff_max=20.0,
    respect_retry_after=True,
    timeout=60.0,
)


def build_questions(targets: Iterable[str] = TARGETS) -> dict[str, Choice]:
    """Create explicit YES/NO/UNK questions for the requested targets."""
    questions = {}
    for target in targets:
        finding = TARGET_DESCRIPTIONS[target]
        questions[target] = Choice(
            instructions={
                "question": f"How does this knee MRI report classify {finding}?",
                "rules": [
                    "Read the report in its original language.",
                    "Use only evidence stated in the report; do not invent findings.",
                    "Choose YES for an explicitly supported abnormal finding.",
                    "Choose NO only when the report explicitly states that the finding is absent or normal.",
                    "Choose UNK when the finding is not addressed, is only a clinical question, or is ambiguous.",
                    "Resolve negation, uncertainty, anatomy, and laterality before answering.",
                ],
            },
            criteria={
                "YES": f"The report explicitly supports {finding}.",
                "NO": f"The report explicitly states that {finding} is absent or normal.",
                "UNK": f"The report does not establish whether there is {finding}.",
            },
        )
    return questions


def label_report(
    client: TypeSafeClient,
    report: str,
    targets: Iterable[str] = TARGETS,
) -> dict[str, Any]:
    """Return tri-state answers and response metadata for one report."""
    requested = list(targets)
    if not requested:
        return {"answers": {}, "model": None}
    if not isinstance(report, str) or not report.strip():
        raise ValueError("Report is empty")

    response = client.system_one(
        state={
            "document_type": "knee MRI radiology report",
            "report": report,
        },
        questions=build_questions(requested),
        retry=RETRY_POLICY,
        timeout=60.0,
    )
    missing = [target for target in requested if target not in response.choices]
    if missing:
        raise ValueError(f"JEV response omitted targets: {missing}")

    answers = {}
    for target in requested:
        answer = response.choices[target]
        verdict = answer.choice.upper()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"Unexpected verdict for {target}: {answer.choice!r}")
        answers[target] = {
            "verdict": verdict,
            "confidence": float(answer.confidence),
            "probabilities": {
                choice.upper(): float(probability)
                for choice, probability in answer.probabilities.items()
            },
        }
    return {"answers": answers, "model": response.model}


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load v2 records while tolerating an interrupted final JSONL write."""
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as checkpoint:
        for line_number, line in enumerate(checkpoint, start=1):
            try:
                record = json.loads(line)
                if record.get("checkpoint_version") != CHECKPOINT_VERSION:
                    raise ValueError("incompatible checkpoint version")
                uid = str(record["StudyInstanceUID"])
                previous = completed.get(uid, {"answers": {}})
                previous["answers"].update(record.get("answers", {}))
                previous.update({k: v for k, v in record.items() if k != "answers"})
                completed[uid] = previous
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                print(f"Ignoring checkpoint line {line_number}: {error}")
    return completed


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"StudyInstanceUID", "Report", *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")


def _seed_lookup(seed_labels: pd.DataFrame | None) -> dict[str, pd.Series]:
    if seed_labels is None:
        return {}
    required = {"StudyInstanceUID", *TARGETS}
    required.update(f"{target}__verdict" for target in TARGETS)
    required.update(f"{target}__conf" for target in TARGETS)
    missing = sorted(required - set(seed_labels.columns))
    if missing:
        raise ValueError(f"Seed labels are missing columns: {missing}")
    if seed_labels["StudyInstanceUID"].duplicated().any():
        raise ValueError("Seed StudyInstanceUID values must be unique")
    return {
        str(row["StudyInstanceUID"]): row
        for _, row in seed_labels.iterrows()
    }


def _unresolved_targets(
    row: pd.Series,
    seed_row: pd.Series | None,
    *,
    audit_all: bool,
) -> list[str]:
    if audit_all:
        return TARGETS.copy()
    if row[TARGETS].notna().all():
        return []
    if seed_row is None:
        return TARGETS.copy()
    return [
        target
        for target in TARGETS
        if str(seed_row[f"{target}__verdict"]).upper() == "UNK"
    ]


def _soft_score(answer: dict[str, Any]) -> float:
    """Expected old-style label score from the JEV choice distribution."""
    probabilities = answer.get("probabilities", {})
    mass = sum(float(probabilities.get(choice, 0.0)) for choice in VALID_VERDICTS)
    if mass <= 0:
        return SCORE_ANCHORS[answer["verdict"]]
    score = sum(
        SCORE_ANCHORS[choice] * float(probabilities.get(choice, 0.0))
        for choice in VALID_VERDICTS
    ) / mass
    return min(1.0, max(0.0, score))


def _apply_results(
    frame: pd.DataFrame,
    seed_labels: pd.DataFrame | None,
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Combine human, v2, and JEV labels as confidence-aware soft scores."""
    output_columns: dict[str, Any] = {}
    seeds = _seed_lookup(seed_labels)

    for target in TARGETS:
        original = frame[target]
        verdicts: list[str] = []
        confidences: list[float] = []
        sources: list[str] = []
        final_values: list[float] = []
        jev_confidences: list[float | None] = []
        jev_verdicts: list[str | None] = []
        jev_probabilities = {choice: [] for choice in VALID_VERDICTS}

        for position, (_, row) in enumerate(frame.iterrows()):
            uid = str(row["StudyInstanceUID"])
            seed = seeds.get(uid)
            jev = results.get(uid, {}).get("answers", {}).get(target)

            if pd.notna(original.iloc[position]):
                value = float(original.iloc[position])
                verdict, confidence, source = ("YES" if value else "NO"), 1.0, "human"
            elif seed is not None and str(seed[f"{target}__verdict"]).upper() in {"YES", "NO"}:
                verdict = str(seed[f"{target}__verdict"]).upper()
                value = float(seed[target])
                confidence = float(seed[f"{target}__conf"])
                source = "report_labels_v2"
            elif jev is not None:
                verdict = jev["verdict"]
                value = _soft_score(jev)
                # Match the old training semantics: a confident abstention is still
                # weak supervision for the abnormality target, not a strong negative.
                confidence = 0.05 if verdict == "UNK" else float(jev["confidence"])
                source = "jev"
            elif seed is not None:
                verdict = str(seed[f"{target}__verdict"]).upper()
                value = float(seed[target])
                confidence = float(seed[f"{target}__conf"])
                source = "report_labels_v2"
            else:
                verdict = "UNK"
                value = SCORE_ANCHORS["UNK"]
                confidence = 0.05
                source = "unknown"

            final_values.append(value)
            verdicts.append(verdict)
            confidences.append(confidence)
            sources.append(source)
            jev_verdicts.append(jev.get("verdict") if jev else None)
            jev_confidences.append(float(jev["confidence"]) if jev else None)
            for choice in VALID_VERDICTS:
                probability = jev.get("probabilities", {}).get(choice) if jev else None
                jev_probabilities[choice].append(probability)

        output_columns[target] = pd.Series(final_values, index=frame.index, dtype="Float64")
        output_columns[f"{target}__verdict"] = verdicts
        output_columns[f"{target}__confidence"] = confidences
        output_columns[f"{target}__conf"] = confidences
        output_columns[f"{target}__source"] = sources
        output_columns[f"{target}__jev_verdict"] = jev_verdicts
        output_columns[f"{target}__jev_confidence"] = jev_confidences
        for choice in sorted(VALID_VERDICTS):
            output_columns[f"{target}__jev_probability_{choice.lower()}"] = jev_probabilities[choice]

    labeled = pd.concat(
        [frame.drop(columns=TARGETS), pd.DataFrame(output_columns, index=frame.index)],
        axis=1,
    )
    labeled["JEVModel"] = labeled["StudyInstanceUID"].map(
        lambda uid: results.get(str(uid), {}).get("model")
    )
    return labeled


def generate_labels(
    frame: pd.DataFrame,
    client: TypeSafeClient,
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_labels: pd.DataFrame | None = None,
    limit: int | None = None,
    workers: int = 4,
    audit_all: bool = False,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Resolve unknown targets with resumable JEV Choice requests."""
    _validate_frame(frame)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(checkpoint_path)
    seeds = _seed_lookup(seed_labels)

    pending: list[tuple[str, str, list[str]]] = []
    eligible_reports = 0
    for _, row in frame.iterrows():
        report = row["Report"]
        if not isinstance(report, str) or not report.strip():
            continue
        uid = str(row["StudyInstanceUID"])
        requested = _unresolved_targets(row, seeds.get(uid), audit_all=audit_all)
        already_done = completed.get(uid, {}).get("answers", {})
        requested = [target for target in requested if target not in already_done]
        if requested:
            eligible_reports += 1
            pending.append((uid, report, requested))
    if limit is not None:
        pending = pending[:limit]

    print(
        f"JEV tri-state labeling: {len(completed)} checkpointed reports, "
        f"{len(pending)} pending, {eligible_reports} reports with unresolved targets"
    )
    errors: list[tuple[str, str]] = []

    def request(item: tuple[str, str, list[str]]) -> tuple[str, dict[str, Any]]:
        uid, report, requested = item
        result = label_report(client, report, requested)
        return uid, {
            "checkpoint_version": CHECKPOINT_VERSION,
            "StudyInstanceUID": uid,
            "requested_targets": requested,
            **result,
        }

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(request, item): item[0] for item in pending}
            for count, future in enumerate(as_completed(futures), start=1):
                uid = futures[future]
                try:
                    _, record = future.result()
                    previous = completed.get(uid, {"answers": {}})
                    previous["answers"].update(record["answers"])
                    previous.update({k: v for k, v in record.items() if k != "answers"})
                    completed[uid] = previous
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                except Exception as error:  # preserve all successful calls
                    errors.append((uid, f"{type(error).__name__}: {error}"))

                if count == 1 or count % 100 == 0 or count == len(pending):
                    print(
                        f"Completed {count}/{len(pending)} reports "
                        f"({len(errors)} errors)"
                    )

    labeled = _apply_results(frame, seed_labels, completed)
    saved_labels = labeled.drop(columns=["Report"], errors="ignore")
    saved_labels.to_csv(output_path, index=False)
    known = int(labeled[TARGETS].notna().sum().sum())
    total = len(labeled) * len(TARGETS)
    print(f"Wrote {output_path} with {known}/{total} known target labels")
    return labeled, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("train.csv"))
    parser.add_argument(
        "--seed-labels", type=Path, default=Path("report_labels_v2.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/train_jev_choice_labeled.csv")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/jev_choice_checkpoint.jsonl"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="Ask JEV about resolved human/v2 labels too, without overwriting them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("TYPESAFE_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = getpass("TypeSafe API key: ")

    frame = pd.read_csv(args.input)
    seed_labels = pd.read_csv(args.seed_labels) if args.seed_labels.exists() else None
    with TypeSafeClient() as client:
        _, errors = generate_labels(
            frame,
            client,
            args.checkpoint,
            args.output,
            seed_labels=seed_labels,
            limit=args.limit,
            workers=args.workers,
            audit_all=args.audit_all,
        )

    if errors:
        print(f"{len(errors)} reports failed; rerun the command to retry them")
        for uid, message in errors[:10]:
            print(f"  {uid}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
