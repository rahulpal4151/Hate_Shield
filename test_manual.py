import os
import sys
from pathlib import Path

# Force UTF-8 for Windows console so Hindi/emoji text prints correctly.
if sys.platform.startswith("win"):
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

# Absolute Project Root Fix
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from utils.inference import load_model, predict_texts

# Paths & Config
MODEL_PATH = PROJECT_ROOT / "models" / "muril_hate_model_v2"
BEST_THRESHOLD = 0.50  # Apne threshold search ka value rakhein

# Load Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"Loading model from: {MODEL_PATH}")

tokenizer, model, device = load_model(MODEL_PATH, device)

def predict_text(text: str):
    prediction = predict_texts(
        [text],
        tokenizer,
        model,
        device,
        threshold=BEST_THRESHOLD,
    )[0]
    result = "HATE 🚨" if prediction.label == "HATE" else "NON-HATE ✅"
    print(f"\nOriginal   : {text}")
    print(f"Cleaned    : {prediction.cleaned_text}")
    print(f"Hate Prob  : {prediction.hate_probability:.4f}")
    print(f"Prediction : {result}")

if __name__ == "__main__":
    test_cases = [
        "आज का दिन बहुत अच्छा है!",
        "मुझे यह खाना बिल्कुल पसंद नहीं आया, बहुत खराब था।",
        "इन लोगों को इस देश से बाहर निकाल देना चाहिए।",
        "तुम सब बेवकूफ और घटिया कौम के लोग हो।"
    ]

    print("\n==============================")
    print("STEP 1: PRE-DEFINED TEST CASES")
    print("==============================")
    for sample in test_cases:
        predict_text(sample)

    print("\n==============================")
    print("STEP 2: LIVE INTERACTIVE TESTING")
    print("==============================")
    print("Sentence type karein (Exit ke liye 'exit' likhein):\n")
    
    while True:
        user_input = input("Enter text: ")
        if user_input.strip().lower() == "exit":
            print("\nTesting completed successfully! 👋")
            break
        if user_input.strip():
            predict_text(user_input)
