from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from .data import DatasetConfig, load_dataframe, prepare_xy

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate phishing detection models.")
    parser.add_argument("--data_path", default=None, help="Optional: path to CSV/TSV/Parquet dataset.")
    parser.add_argument("--text_col", default="text", help="Text column name.")
    parser.add_argument("--label_col", default="label", help="Label column name.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    cfg = DatasetConfig(data_path=args.data_path, text_col=args.text_col, label_col=args.label_col)
    df = load_dataframe(cfg)
    x, y = prepare_xy(df, cfg.text_col, cfg.label_col)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    models = {
        "naive_bayes": MultinomialNB(),
        "log_reg": LogisticRegression(max_iter=2000),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_test, preds).tolist()

        metrics[name] = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, preds, zero_division=0),
        }

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best_name = max(metrics.keys(), key=lambda k: metrics[k]["f1"])
    cm = np.array(metrics[best_name]["confusion_matrix"])

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix ({best_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit(0)", "Phishing(1)"])
    ax.set_yticklabels(["Legit(0)", "Phishing(1)"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("Done. Metrics saved to outputs/metrics.json")
    print(f"Best model by F1: {best_name}")
    print(metrics[best_name]["classification_report"])

if __name__ == "__main__":
    main()
