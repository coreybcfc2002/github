# AI-Based Phishing Detection (NLP + ML)

This repository contains a simple, reproducible phishing detection pipeline using **TF-IDF** + classic supervised ML models (**Naive Bayes** and **Logistic Regression**).

## What it does
- Loads a labelled phishing dataset (default: Hugging Face `ealvaradob/phishing-dataset`)
- Cleans/preprocesses text
- Vectorises text using TF-IDF
- Trains Naive Bayes + Logistic Regression
- Evaluates using Accuracy / Precision / Recall / F1 + confusion matrix
- Saves trained models to `models/`

## Quick start

### 1) Create environment + install requirements
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run training + evaluation
```bash
python -m code.train_and_evaluate
```

Outputs:
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `models/tfidf_vectorizer.joblib`
- `models/naive_bayes.joblib`
- `models/log_reg.joblib`

### 3) Predict on your own text
```bash
python -m code.predict --text "Your account is locked. Verify now: http://example.com"
```

## Dataset
Default loader uses Hugging Face datasets:
- **ealvaradob/phishing-dataset**

If you prefer a local file, you can provide:
```bash
python -m code.train_and_evaluate --data_path path/to/your_dataset.csv --text_col text --label_col label
```

Your CSV should contain:
- a text column (e.g., `text`)
- a label column with values like `phishing/spam` vs `ham/legitimate` (binary)
