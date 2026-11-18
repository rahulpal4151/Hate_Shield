import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inference import load_model, predict_texts


MODEL_DIR = PROJECT_ROOT / "models" / "muril_hate_model_v2"
TEST_FILE = PROJECT_ROOT / "data" / "raw" / "SubTask-B-test.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "holdout_metrics.json"
DEFAULT_THRESHOLD = 0.50


def main() -> Dict[str, object]:
    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"Labeled test CSV not found: {TEST_FILE}")

    test_df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(test_df.columns)
    if missing_columns:
        raise ValueError(
            "Test CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if test_df.empty or test_df["text"].isna().any() or test_df["label"].isna().any():
        raise ValueError("Test CSV must contain non-empty text and label values.")

    numeric_labels = pd.to_numeric(test_df["label"], errors="raise")
    if not numeric_labels.isin([0, 1]).all():
        raise ValueError("Test labels must contain only 0 and 1.")
    labels = numeric_labels.astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Test labels must contain both classes 0 and 1 only.")

    tokenizer, model, device = load_model(MODEL_DIR)
    predictions = predict_texts(
        test_df["text"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        threshold=DEFAULT_THRESHOLD,
    )
    y_true = labels.tolist()
    y_pred = [1 if prediction.label == "HATE" else 0 for prediction in predictions]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics: Dict[str, object] = {
        "dataset": TEST_FILE.name,
        "samples": len(test_df),
        "threshold": DEFAULT_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }
    OUTPUT_FILE.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nMetrics saved to: {OUTPUT_FILE}")
    return metrics


if __name__ == "__main__":
    main()
