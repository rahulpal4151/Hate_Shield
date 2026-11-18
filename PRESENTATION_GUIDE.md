# HateShield presentation guide

## Suggested 5-slide flow

1. **Problem** — Harmful language in Hindi/Hinglish can be difficult to flag
   consistently with keyword rules alone.
2. **Solution** — HateShield fine-tunes MuRIL, a multilingual transformer, for
   binary text classification: HATE (1) or NON-HATE (0).
3. **Workflow** — Clean and balance training examples, tokenize, fine-tune,
   select an operating threshold on validation data, and evaluate on a separate
   labeled test split.
4. **Live demo** — Open the app, try one neutral sentence, show the hate
   probability, then upload a CSV and download the predictions.
5. **Results and limitations** — Run the holdout evaluator before the talk and
   show its actual precision, recall, F1, accuracy, and confusion matrix.
   Explain that context, sarcasm, dialect, and label quality remain challenges.

## Before the presentation

1. Start the app using the Windows commands in [README.md](./README.md).
2. Run `.\venv\Scripts\python.exe evaluation\evaluate_holdout.py`.
3. Confirm the metrics shown in the app match
   `evaluation/holdout_metrics.json`.
4. Test single-text prediction and CSV upload/download once before presenting.
5. Keep the saved model directory available; the demo loads it locally.

Do not claim a metric unless it was produced by the held-out evaluation for the
same model version being demonstrated. The displayed probability is a model
score and is not guaranteed to be calibrated.
