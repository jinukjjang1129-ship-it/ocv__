"""Four-model early OCV experiment with train/validation/test isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .features import build_features, encode_target


def split_indices(
    frame: pd.DataFrame,
    target: pd.Series,
    strategy: str = "temporal",
    time_column: str = "생산일자",
    group_column: str | None = None,
) -> dict[str, np.ndarray]:
    """Keep date ties together; reject overlapping groups in temporal partitions."""
    indices = np.arange(len(frame))
    if strategy == "temporal":
        dates = pd.to_datetime(frame[time_column], errors="raise")
        if dates.isna().any():
            raise ValueError("Production dates must not be missing")
        unique = sorted(dates.unique())
        if len(unique) < 5:
            raise ValueError("At least 5 distinct production dates are required")
        reserve = max(1, math.ceil(len(unique) * 0.2))
        first, second = unique[-2 * reserve], unique[-reserve]
        parts = {
            "train": indices[dates < first],
            "validation": indices[(dates >= first) & (dates < second)],
            "test": indices[dates >= second],
        }
    elif strategy == "group":
        if not group_column or frame[group_column].isna().any():
            raise ValueError("Group split requires a complete group column")
        groups = frame[group_column]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        train, remainder = next(splitter.split(frame, target, groups))
        val_local, test_local = next(
            GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(
                frame.iloc[remainder], target.iloc[remainder], groups.iloc[remainder]
            )
        )
        parts = {
            "train": train,
            "validation": remainder[val_local],
            "test": remainder[test_local],
        }
    elif strategy == "random":
        train, remainder = train_test_split(
            indices, test_size=0.4, stratify=target, random_state=42
        )
        validation, test = train_test_split(
            remainder, test_size=0.5, stratify=target.iloc[remainder], random_state=42
        )
        parts = {"train": train, "validation": validation, "test": test}
    else:
        raise ValueError("Unknown splitting strategy")
    for name, positions in parts.items():
        if len(positions) == 0 or target.iloc[positions].nunique() != 2:
            raise ValueError(f"Both classes are required in {name}")
    if group_column:
        seen: set = set()
        for positions in parts.values():
            groups = set(frame.iloc[positions][group_column])
            if seen & groups:
                raise ValueError("A cell/batch group spans partitions")
            seen |= groups
    return parts


def make_models(target_train: pd.Series) -> dict[str, Pipeline]:
    """Use only training class frequencies; hold model budgets fixed across splits."""
    imbalance = float((target_train == 0).sum() / (target_train == 1).sum())
    classifiers = {
        "logistic": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, min_samples_leaf=10, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=2,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            scale_pos_weight=imbalance,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=2,
            random_state=42,
        ),
    }
    return {
        name: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler() if name == "logistic" else "passthrough"),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def choose_threshold(target: pd.Series, scores: np.ndarray) -> float:
    """Select F1 on a prespecified validation grid, never on the test set."""
    candidates = np.arange(0.05, 1, 0.05)
    return float(
        max(
            candidates,
            key=lambda t: (
                f1_score(target, scores >= t, zero_division=0),
                precision_score(target, scores >= t, zero_division=0),
                t,
            ),
        )
    )


def metrics(target: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    precision, recall, _ = precision_recall_curve(target, scores)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": len(target),
        "prevalence": float(target.mean()),
        "precision": precision_score(target, predicted, zero_division=0),
        "recall": recall_score(target, predicted, zero_division=0),
        "f1": f1_score(target, predicted, zero_division=0),
        "average_precision": average_precision_score(target, scores),
        "pr_auc_trapezoid": auc(recall[::-1], precision[::-1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "escape_per_unit": float(fn / len(target)),
        "false_reject_per_unit": float(fp / len(target)),
    }


def experiment(
    frame: pd.DataFrame,
    output: Path,
    strategy: str = "temporal",
    time_column: str = "생산일자",
    group_column: str | None = None,
) -> dict[str, Any]:
    """Write aggregate results and private error cases, returning fitted models."""
    frame = frame.reset_index(drop=True)
    target = encode_target(frame.OK_NG)
    features = build_features(frame)
    partitions = split_indices(frame, target, strategy, time_column, group_column)
    train, validation, test = (partitions[n] for n in ["train", "validation", "test"])
    if features.iloc[train].isna().all().any():
        raise ValueError("A training feature is entirely missing")
    models = make_models(target.iloc[train])
    results: dict[str, dict] = {}
    scores_by_model = {}
    validation_ap = {}
    thresholds = {}
    for name, model in models.items():
        model.fit(features.iloc[train], target.iloc[train])
        val_scores = model.predict_proba(features.iloc[validation])[:, 1]
        thresholds[name] = choose_threshold(target.iloc[validation], val_scores)
        validation_ap[name] = average_precision_score(
            target.iloc[validation], val_scores
        )
        results[name] = {
            "validation": metrics(target.iloc[validation], val_scores, thresholds[name])
        }
    # Select before accessing test scores. Test comparisons are descriptive only.
    selected = max(validation_ap, key=validation_ap.get)
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        scores = model.predict_proba(features.iloc[test])[:, 1]
        scores_by_model[name] = scores
        results[name]["test_default"] = metrics(target.iloc[test], scores, 0.5)
        results[name]["test_selected_threshold"] = metrics(
            target.iloc[test], scores, thresholds[name]
        )
        precision, recall, _ = precision_recall_curve(target.iloc[test], scores)
        axis.plot(
            recall,
            precision,
            label=f"{name} (AP={results[name]['test_default']['average_precision']:.3f})",
        )
        predictions = scores >= thresholds[name]
        errors = frame.iloc[test].copy()
        errors["row_position"] = test
        errors["ng_score"] = scores
        errors["predicted_ng"] = predictions
        errors["error_type"] = np.where(
            (target.iloc[test].to_numpy() == 1) & ~predictions,
            "FN",
            np.where(
                (target.iloc[test].to_numpy() == 0) & predictions, "FP", "correct"
            ),
        )
        errors.loc[errors.error_type != "correct"].to_csv(
            private / f"{name}_errors.csv", index=False
        )
    axis.axhline(
        target.iloc[test].mean(), color="gray", linestyle="--", label="Prevalence"
    )
    axis.set(
        xlabel="Recall", ylabel="Precision", title="OCV1–3: held-out production dates"
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=160)
    plt.close(fig)
    chosen_scores = scores_by_model[selected]
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        target.iloc[test],
        chosen_scores >= thresholds[selected],
        labels=[0, 1],
        display_labels=["OK", "NG"],
        ax=axis,
        colorbar=False,
    )
    axis.set_title(
        f"Validation-selected {selected}; threshold={thresholds[selected]:.2f}"
    )
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    sweep = [
        metrics(
            target.iloc[validation],
            models[selected].predict_proba(features.iloc[validation])[:, 1],
            float(t),
        )
        for t in np.arange(0.05, 1, 0.05)
    ]
    pd.DataFrame(sweep).to_csv(output / "validation_thresholds.csv", index=False)
    report = {
        "dataset_role": "provided teaching sample; manufacturing provenance unverified",
        "split": strategy,
        "seed": 42,
        "feature_columns": list(features.columns),
        "selection_rule": "maximum validation average precision; validation-grid F1 threshold",
        "selected_model": selected,
        "partitions": {
            name: {
                "rows": len(indices),
                "ng": int(target.iloc[indices].sum()),
                "ok": int((target.iloc[indices] == 0).sum()),
                "production_dates": int(frame.iloc[indices][time_column].nunique())
                if time_column in frame
                else None,
            }
            for name, indices in partitions.items()
        },
        "quality": {
            "rows": len(frame),
            "ng": int(target.sum()),
            "prevalence": float(target.mean()),
            "missing_inputs": features[list(("OCV1", "OCV2", "OCV3"))]
            .isna()
            .sum()
            .to_dict(),
            "duplicate_rows": int(frame.duplicated().sum()),
            "target_matches_ocv4_lt35": bool(
                (target == (pd.to_numeric(frame.OCV4, errors="coerce") < 35)).all()
            )
            if "OCV4" in frame
            else None,
        },
        "models": results,
        "limitations": [
            "Teaching sample is not verified production/customer data",
            "Measurement intervals and physical units are unconfirmed",
            "No cell ID; repeated-cell leakage cannot be fully excluded",
            "Only a few production dates and one product in supplied sample",
            "Probability scores are not calibrated",
            "Test comparisons are retrospective; new future data is required",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez(private / "split_indices.npz", **partitions)
    return {
        "report": report,
        "model": models[selected],
        "threshold": thresholds[selected],
        "features": list(features.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--header", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["temporal", "group", "random"], default="temporal"
    )
    parser.add_argument("--time-column", default="생산일자")
    parser.add_argument("--group-column")
    args = parser.parse_args()
    frame = pd.read_excel(args.input, header=args.header)
    frame.columns = frame.columns.astype(str).str.strip()
    result = experiment(
        frame, args.output, args.split, args.time_column, args.group_column
    )
    result["report"]["input_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    (args.output / "metrics.json").write_text(json.dumps(result["report"], indent=2))
    print(
        json.dumps(
            {
                "selected_model": result["report"]["selected_model"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
