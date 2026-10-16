"""
train_model.py - trains the SpamGuard AI spam classifier.

What this script does (step by step):
  1. Finds (or downloads) the SMS Spam Collection dataset.
  2. Cleans it: removes missing values, bad labels and duplicate messages.
  3. Splits it into a TRAINING set and a TESTING set.
  4. Trains a Pipeline = TF-IDF Vectorizer + Multinomial Naive Bayes.
     Because both steps live in ONE pipeline and the pipeline is fitted on the
     training data only, the vectorizer never sees the test data (no data leakage).
  5. Evaluates the model on the unseen test set (accuracy, precision, recall,
     F1-score, confusion matrix).
  6. Saves the pipeline + the real metrics to models/spam_classifier.joblib.

Usage:
    python train_model.py --download        # download the dataset, then train
    python train_model.py                   # use a dataset already in data/
    python train_model.py --data path/to/file.csv
"""

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"

# Official UCI download link for the SMS Spam Collection dataset.
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# File names we look for inside the data/ folder (first match wins).
CANDIDATE_FILES = [
    "SMSSpamCollection",
    "SMSSpamCollection.tsv",
    "SMSSpamCollection.txt",
    "spam.csv",
    "sms_spam.csv",
]

RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Step 1: getting the dataset
# ----------------------------------------------------------------------------
def download_dataset() -> Path:
    """Download the UCI SMS Spam Collection zip and extract it into data/."""
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading dataset from:\n  {DATASET_URL}")
    try:
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extract("SMSSpamCollection", DATA_DIR)
    except Exception as error:  # network problems, bad zip, missing member...
        print("\n[ERROR] Could not download the dataset automatically.")
        print(f"        Reason: {error}")
        print_manual_download_help()
        sys.exit(1)
    path = DATA_DIR / "SMSSpamCollection"
    print(f"Dataset saved to: {path}")
    return path


def print_manual_download_help() -> None:
    print(
        "\nManual setup (takes 1 minute):\n"
        "  1. Open https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
        "     and click 'Download' (or search 'SMS Spam Collection' on Kaggle).\n"
        "  2. Unzip it and copy the file 'SMSSpamCollection' (or 'spam.csv')\n"
        f"     into this folder: {DATA_DIR}\n"
        "  3. Run:  python train_model.py\n"
    )


def find_dataset(explicit_path):
    """Return the dataset path, or None if nothing is found."""
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    for name in CANDIDATE_FILES:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------------------
# Step 2: loading and cleaning
# ----------------------------------------------------------------------------
def read_file_text(path: Path) -> str:
    """Read a text file, trying UTF-8 first and then Latin-1 (older Kaggle files)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load either the UCI tab-separated file or a Kaggle-style CSV."""
    text = read_file_text(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(io.StringIO(text))
        # Different CSV versions use different column names.
        renames = {"v1": "label", "v2": "text", "category": "label",
                   "message": "text", "class": "label", "sms": "text"}
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns=renames)
    else:
        # UCI format: one message per line -> label<TAB>message
        df = pd.read_csv(io.StringIO(text), sep="\t", header=None,
                         names=["label", "text"], quoting=csv.QUOTE_NONE)

    if not {"label", "text"}.issubset(df.columns):
        raise ValueError(
            "Could not find 'label' and 'text' columns in the dataset. "
            f"Columns found: {list(df.columns)}"
        )
    return df[["label", "text"]]


def clean_dataset(df: pd.DataFrame):
    """Handle missing values, labels and duplicates. Returns (clean_df, report)."""
    report = {"rows_loaded": int(len(df))}

    df = df.dropna(subset=["label", "text"]).copy()
    report["missing_removed"] = report["rows_loaded"] - len(df)

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    before = len(df)
    df = df[df["label"].isin(["ham", "spam"]) & (df["text"] != "")]
    report["invalid_removed"] = before - len(df)

    # Duplicates would let the same message appear in both train and test sets,
    # which makes the scores look better than they really are.
    before = len(df)
    df = df.drop_duplicates(subset="text")
    report["duplicates_removed"] = before - len(df)

    df["target"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    report["rows_final"] = int(len(df))
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------
# Step 3-5: training and evaluating
# ----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """TF-IDF turns text into numbers; Naive Bayes learns spam vs ham from them."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   # single words and word pairs
            sublinear_tf=True,
            min_df=2,             # ignore words seen in only one message
        )),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2):
    """Split, fit on the training set only, evaluate on the test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["target"],   # keep the spam/ham ratio equal in both sets
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)          # vectorizer is fitted on TRAIN data only

    predictions = pipeline.predict(X_test)  # test data is only transformed, never fitted
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts_total": {"ham": int((df["target"] == 0).sum()),
                               "spam": int((df["target"] == 1).sum())},
        "class_counts_train": {"ham": int((y_train == 0).sum()),
                               "spam": int((y_train == 1).sum())},
        "class_counts_test": {"ham": int((y_test == 0).sum()),
                              "spam": int((y_test == 1).sum())},
    }
    return pipeline, metrics


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SpamGuard AI model.")
    parser.add_argument("--download", action="store_true",
                        help="download the UCI SMS Spam Collection into data/")
    parser.add_argument("--data", help="path to a dataset file (UCI tab file or CSV)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction of data used for testing (default 0.2)")
    parser.add_argument("--output", default=str(MODEL_PATH),
                        help="where to save the model (default models/spam_classifier.joblib)")
    args = parser.parse_args()

    # 1. Dataset
    data_path = download_dataset() if args.download else find_dataset(args.data)
    if data_path is None:
        print("[ERROR] No dataset found.")
        print_manual_download_help()
        print("Or let the script download it for you:  python train_model.py --download")
        sys.exit(1)
    print(f"Using dataset: {data_path}")

    # 2. Load + clean
    try:
        raw = load_raw_dataset(data_path)
    except Exception as error:
        print(f"[ERROR] Could not read the dataset: {error}")
        sys.exit(1)
    df, report = clean_dataset(raw)
    print(f"Rows loaded: {report['rows_loaded']} | missing removed: {report['missing_removed']} | "
          f"invalid labels removed: {report['invalid_removed']} | "
          f"duplicates removed: {report['duplicates_removed']} | rows used: {report['rows_final']}")

    if df["target"].nunique() < 2 or len(df) < 50:
        print("[ERROR] The dataset must contain both 'spam' and 'ham' messages (at least ~50 rows).")
        sys.exit(1)

    # 3-5. Train + evaluate
    pipeline, metrics = train_and_evaluate(df, test_size=args.test_size)

    print("\n--- Test-set results (data the model never saw during training) ---")
    print(f"Training samples : {metrics['n_train']}")
    print(f"Testing samples  : {metrics['n_test']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision (spam) : {metrics['precision']:.4f}")
    print(f"Recall (spam)    : {metrics['recall']:.4f}")
    print(f"F1-score (spam)  : {metrics['f1']:.4f}")
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print(f"Confusion matrix : TN={tn} FP={fp} FN={fn} TP={tp}")

    # 6. Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "metrics": metrics,
        "meta": {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset_file": data_path.name,
            "cleaning_report": report,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
    }
    joblib.dump(bundle, output, compress=3)
    print(f"\nModel saved to: {output}")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
