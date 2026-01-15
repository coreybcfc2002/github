from __future__ import annotations
import argparse
from pathlib import Path
import joblib

from .utils import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict phishing vs legitimate for a single text.")
    parser.add_argument("--text", required=True, help="Input message text")
    parser.add_argument("--model", default="log_reg", choices=["log_reg", "naive_bayes"])
    args = parser.parse_args()

    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / f"{args.model}.joblib")

    cleaned = clean_text(args.text)
    X = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])

    print("PHISHING/SPAM (1)" if pred == 1 else "LEGIT/HAM (0)")

if __name__ == "__main__":
    main()
