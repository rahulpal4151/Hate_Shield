# HateShield

HateShield is a student project that classifies Hindi/Hinglish text as
**HATE** or **NON-HATE** using a fine-tuned Google MuRIL sequence-classification
model. It includes a Streamlit demo, a CSV batch-prediction workflow, training
and evaluation scripts, and a held-out test-set evaluator.

> This is a prototype for learning and demonstration. It can miss context,
> sarcasm, dialect, or spelling variation; predictions should not be used as
> automatic moderation decisions without human review.

## Run the demo (Windows)

From the project root, activate the existing environment and start Streamlit:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Or, without activating the environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The demo expects the trained model files in
`models/muril_hate_model_v2/`. It runs locally and loads the model from disk.

### Demo features

- Classify a single Hindi/Hinglish sentence.
- Upload a CSV with a `text` or `tweet` column, preview predictions, and
  download the full results.
- Adjust the hate-probability threshold for demonstration.
- View held-out metrics after running the evaluation script.

## Reproduce the held-out evaluation

The project contains an included labeled held-out test set at
`data/raw/SubTask-B-test.csv`. Evaluate the saved model at the default 0.50
threshold:

```powershell
.\venv\Scripts\python.exe evaluation\evaluate_holdout.py
```

The script reports accuracy, hate-class precision/recall/F1, and a confusion
matrix, then writes the results to `evaluation/holdout_metrics.json`. The test
set is evaluated at a fixed threshold; use the validation set for any threshold
selection, not the held-out test set. Do not quote metrics until this command
has completed for the model being presented.

## Project workflow

1. `preprocessing/text_cleaning.py` cleans text and removes duplicate or empty
   examples.
2. `preprocessing/balance_dataset.py` balances the training data.
3. `dataset/build_dataset.py` tokenizes and creates train/validation datasets.
4. `training/train_model.py` fine-tunes MuRIL.
5. `evaluation/threshold_tuning.py` explores thresholds on validation data.
6. `evaluation/evaluate_holdout.py` measures the saved model on the separate
   labeled test CSV.
7. `app.py` provides interactive single-text and CSV demos.

Run the pipeline scripts from the project root, in order, when retraining.
Training downloads `google/muril-base-cased` if it is not already cached and
requires substantial memory and time. The existing locally saved model is used
for the presentation demo.

## Tests

Run the lightweight inference-helper tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a slide outline and pre-presentation checklist, see
[PRESENTATION_GUIDE.md](./PRESENTATION_GUIDE.md).

## Dependencies

The workspace environment already contains the packages used by the app and
training scripts. To create another environment, install `requirements.txt`
first. PyTorch installation can vary by operating system and CUDA version; use
the official PyTorch installation selector if the generic install is unsuitable
for your machine.

## Data and model notes

- Labels: `0` = NON-HATE, `1` = HATE.
- Backbone: `google/muril-base-cased`, fine-tuned for two-class
  sequence classification.
- The processed training CSV has 18,179 examples; the balanced training file
  has 6,018 examples (4,012 NON-HATE and 2,006 HATE).
- The included labeled test CSV has 4,076 examples (3,602 NON-HATE and
  474 HATE). It is evaluated separately from the training/validation pipeline.
- The default presentation threshold is `0.50`; it is a configurable operating
  point, not a guarantee of calibrated probability.
- The test data comes from the included labeled test CSV. Verify the resulting
  metrics before presenting them.
