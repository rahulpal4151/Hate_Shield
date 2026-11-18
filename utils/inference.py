import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class Prediction:
    cleaned_text: str
    label: str
    hate_probability: float


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(
    model_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


def predict_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    threshold: float = 0.5,
    max_length: int = 128,
    batch_size: int = 32,
) -> List[Prediction]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")

    cleaned_texts = [clean_text(text) for text in texts]
    predictions: List[Prediction] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            model_inputs = {
                name: value.to(device) for name, value in inputs.items()
            }
            probabilities = torch.softmax(
                model(**model_inputs).logits,
                dim=1,
            )[:, 1].cpu().tolist()

            predictions.extend(
                Prediction(
                    cleaned_text=text,
                    label="HATE" if probability >= threshold else "NON-HATE",
                    hate_probability=float(probability),
                )
                for text, probability in zip(batch_texts, probabilities)
            )

    return predictions
