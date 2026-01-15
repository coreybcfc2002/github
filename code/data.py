from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from .utils import clean_text, coerce_label

@dataclass
class DatasetConfig:
    data_path: Optional[str] = None
    text_col: str = "text"
    label_col: str = "label"

def load_dataframe(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Load dataset into a DataFrame.

    - If cfg.data_path is None: load Hugging Face dataset 'ealvaradob/phishing-dataset'
    - Else: load from CSV/TSV/Parquet based on extension.
    """
    if cfg.data_path is None:
        from datasets import load_dataset  # pip install datasets
        ds = load_dataset("ealvaradob/phishing-dataset")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()

        # Auto-detect if names differ
        if cfg.text_col not in df.columns:
            for c in ["text", "body", "content", "message"]:
                if c in df.columns:
                    cfg.text_col = c
                    break
        if cfg.label_col not in df.columns:
            for c in ["label", "class", "category", "target"]:
                if c in df.columns:
                    cfg.label_col = c
                    break
        return df

    p = cfg.data_path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(cfg.data_path)
    if p.endswith(".tsv"):
        return pd.read_csv(cfg.data_path, sep="\\t")
    if p.endswith(".parquet"):
        return pd.read_parquet(cfg.data_path)
    raise ValueError("Unsupported file type. Use .csv, .tsv, or .parquet")

def prepare_xy(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series]:
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Missing required columns. Found columns: {list(df.columns)}")
    x = df[text_col].astype(str).map(clean_text)
    y = df[label_col].map(coerce_label)
    mask = x.str.len() > 0
    return x[mask], y[mask]
