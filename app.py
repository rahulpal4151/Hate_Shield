import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.inference import load_model, predict_texts


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "muril_hate_model_v2"
METRICS_FILE = PROJECT_ROOT / "evaluation" / "holdout_metrics.json"

st.set_page_config(
    page_title="HateShield | Hindi Hate-Speech Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HateShield")
st.subheader("Hindi Hate-Speech Classification")
st.caption(
    "MuRIL-based text classification demo • Predictions are assistive, "
    "not a substitute for human review."
)

if not MODEL_DIR.is_dir():
    st.error(
        f"Trained model not found at `{MODEL_DIR}`. "
        "Restore the model folder before starting the demo."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading the fine-tuned MuRIL model...")
def get_model():
    return load_model(MODEL_DIR)


with st.sidebar:
    st.header("Demo settings")
    threshold = st.slider(
        "Hate decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help=(
            "A higher threshold is more conservative about flagging text. "
            "Changing it affects predictions, not the underlying model."
        ),
    )
    st.caption("Model: fine-tuned Google MuRIL base")
    st.caption("Classes: NON-HATE (0) and HATE (1)")

single_tab, batch_tab, about_tab = st.tabs(
    ["Single text", "CSV batch", "Model & evaluation"]
)

with single_tab:
    st.markdown("Enter a Hindi or Hinglish sentence to classify.")
    text = st.text_area(
        "Text to check",
        placeholder="Yahan apna Hindi ya Hinglish text likhein...",
        height=120,
    )
    if st.button("Classify text", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            tokenizer, model, device = get_model()
            result = predict_texts(
                [text],
                tokenizer,
                model,
                device,
                threshold=threshold,
            )[0]
            if result.label == "HATE":
                st.error(f"Prediction: {result.label}")
            else:
                st.success(f"Prediction: {result.label}")
            st.metric("Hate probability", f"{result.hate_probability:.1%}")
            with st.expander("See normalized text"):
                st.write(result.cleaned_text or "(empty after cleaning)")

with batch_tab:
    st.markdown(
        "Upload a UTF-8 CSV with a `text` column (or a `tweet` column). "
        "The original columns are kept and predictions are appended."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except (
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            st.error(f"Could not read this CSV: {error}")
        else:
            text_column = "text" if "text" in uploaded_df.columns else "tweet"
            if text_column not in uploaded_df.columns:
                st.error("CSV must contain a `text` or `tweet` column.")
            elif uploaded_df.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                if st.button("Classify CSV", type="primary"):
                    texts = uploaded_df[text_column].fillna("").astype(str).tolist()
                    tokenizer, model, device = get_model()
                    with st.spinner(f"Classifying {len(texts):,} rows..."):
                        results = predict_texts(
                            texts,
                            tokenizer,
                            model,
                            device,
                            threshold=threshold,
                        )
                    result_df = uploaded_df.copy()
                    result_df["clean_text"] = [
                        result.cleaned_text for result in results
                    ]
                    result_df["prediction"] = [
                        result.label for result in results
                    ]
                    result_df["hate_probability"] = [
                        result.hate_probability for result in results
                    ]
                    st.success(f"Classified {len(result_df):,} rows.")
                    st.dataframe(result_df.head(100), use_container_width=True)
                    st.download_button(
                        "Download all predictions",
                        data=result_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="hateshield_predictions.csv",
                        mime="text/csv",
                    )

with about_tab:
    st.markdown(
        """
        **Pipeline:** Hindi/Hinglish text → text normalization → MuRIL tokenizer
        → fine-tuned transformer classifier → hate probability and label.

        **Responsible-use note:** This is a student prototype, not a moderation
        decision system. Sarcasm, context, dialect, spelling variation, and
        identity-related language can affect predictions. A person should
        review flagged content.
        """
    )
    if METRICS_FILE.is_file():
        with METRICS_FILE.open(encoding="utf-8") as metrics_stream:
            metrics = json.load(metrics_stream)
        st.markdown("#### Held-out test-set results")
        st.caption(
            f"{metrics['samples']:,} labeled samples • "
            f"decision threshold {metrics['threshold']:.2f}"
        )
        columns = st.columns(4)
        for column, metric_name in zip(
            columns,
            ["accuracy", "precision", "recall", "f1"],
        ):
            column.metric(metric_name.upper(), f"{metrics[metric_name]:.1%}")
        st.markdown(
            "Confusion matrix (rows = actual, columns = predicted; "
            "NON-HATE then HATE)"
        )
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual NON-HATE", "Actual HATE"],
                columns=["Predicted NON-HATE", "Predicted HATE"],
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "Run `python evaluation/evaluate_holdout.py` to calculate "
            "held-out results for this model."
        )
