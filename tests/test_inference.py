import unittest
from types import SimpleNamespace

import torch

from utils.inference import clean_text, predict_texts


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_texts = []

    def __call__(self, texts, **kwargs):
        self.last_texts = texts
        return {
            "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        logits = torch.tensor([[0.0, 2.0]] * len(kwargs["input_ids"]))
        return SimpleNamespace(logits=logits)


class InferenceTests(unittest.TestCase):
    def test_clean_text_removes_urls_mentions_and_extra_whitespace(self):
        self.assertEqual(
            clean_text(" नमस्ते  @user https://example.com <b>दुनिया</b> "),
            "नमस्ते दुनिया",
        )

    def test_predict_texts_applies_threshold_and_preserves_cleaned_text(self):
        tokenizer = FakeTokenizer()
        predictions = predict_texts(
            ["@user नमस्ते"],
            tokenizer,
            FakeModel(),
            torch.device("cpu"),
            threshold=0.8,
        )

        self.assertEqual(tokenizer.last_texts, ["नमस्ते"])
        self.assertEqual(predictions[0].label, "HATE")
        self.assertEqual(predictions[0].cleaned_text, "नमस्ते")
        self.assertGreater(predictions[0].hate_probability, 0.8)

    def test_predict_texts_rejects_out_of_range_threshold(self):
        with self.assertRaises(ValueError):
            predict_texts(
                [],
                FakeTokenizer(),
                FakeModel(),
                torch.device("cpu"),
                threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
