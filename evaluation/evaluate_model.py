from pathlib import Path

import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


# Resolve paths from this file so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "muril_hate_model_v2"
VAL_DATA_DIR = PROJECT_ROOT / "data" / "tokenized" / "val"
BATCH_SIZE = 16
LABELS = [0, 1]
LABEL_NAMES = ["Non-Hate", "Hate"]


def main() -> None:
    print("\n==============================")
    print("STEP 6: MODEL EVALUATION")
    print("==============================")

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not VAL_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Validation dataset not found: {VAL_DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading validation dataset...")
    val_dataset = load_from_disk(str(VAL_DATA_DIR))
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {VAL_DATA_DIR}")
    required_columns = {"input_ids", "attention_mask", "label"}
    missing_columns = required_columns.difference(val_dataset.column_names)
    if missing_columns:
        raise ValueError(
            "Validation dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    print("Validation samples:", len(val_dataset))
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    print("\nRunning inference on validation set...")

    with torch.no_grad():
        for batch in loader:
            model_inputs = {
                "input_ids": batch["input_ids"].long().to(device),
                "attention_mask": batch["attention_mask"].long().to(device),
            }
            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"].long().to(device)

            logits = model(**model_inputs).logits
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_true.extend(batch["label"].cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix (rows=true, columns=predicted; Non-Hate, Hate):")
    print(cm)
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    print("Evaluation completed successfully.")
    print("==============================\n")


if __name__ == "__main__":
    main()
