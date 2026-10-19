"""Download, clean, deduplicate and split the SMS phishing dataset.

Dataset: Mishra & Soni, "SMS Phishing Dataset for Machine Learning and Pattern Recognition",
Mendeley Data, DOI 10.17632/f45bkkt8pr.1, CC BY 4.0.

Two kinds of duplication are handled differently:

* Exact duplicates (same text) are dropped. If the same text carries two different labels,
  every copy is dropped, because the ground truth is ambiguous.
* Template variants (same text except for the link, amount or code) are kept, but grouped.
  Scam campaigns send one template to many people with a different short link each time.
  If one variant lands in train and another in test, the test score measures memorisation,
  not generalisation. Splitting by a normalised "template key" keeps every variant of a
  template in the same split.

Run ``python -m smishing.data`` to download the dataset and write data/splits/*.csv.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from smishing import LABELS

DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_ZIP = DATA_DIR / "raw" / "Dataset_5971.zip"
SPLITS_DIR = DATA_DIR / "splits"

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"  # explicit links
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,6}/\S*"  # shortened links such as bit.do/abc
    r"|\b[a-z0-9-]+\.(?:com|net|org|io|co|uk|info|biz)\b",  # bare domains
    re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


def template_key(text: str) -> str:
    """Normalise a message so that variants of one template map to the same key."""
    key = text.lower()
    key = _URL_RE.sub(" <url> ", key)
    key = _DIGITS_RE.sub("0", key)
    return _SPACE_RE.sub(" ", key).strip()


def download(dest: Path = RAW_ZIP) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_raw(path: Path = RAW_ZIP) -> pd.DataFrame:
    """Return the dataset as a frame with columns ``text`` and ``label``."""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("Dataset_5971.csv")
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    df = df.rename(columns={"TEXT": "text", "LABEL": "label"})[["text", "label"]]
    df["label"] = df["label"].str.strip().str.lower()
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected labels in dataset: {sorted(unknown)}")
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


@dataclass
class DedupeReport:
    rows_in: int
    exact_duplicates_dropped: int
    conflicting_texts: int
    conflicting_rows_dropped: int
    rows_out: int


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupeReport]:
    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    is_conflicting = df["text"].isin(conflicting)

    clean = df[~is_conflicting]
    deduped = clean.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    report = DedupeReport(
        rows_in=len(df),
        exact_duplicates_dropped=len(clean) - len(deduped),
        conflicting_texts=len(conflicting),
        conflicting_rows_dropped=int(is_conflicting.sum()),
        rows_out=len(deduped),
    )
    return deduped, report


def split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Stratified split grouped by template key, roughly 72/14/14 train/val/test."""
    df = df.assign(key=df["text"].map(template_key))

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["label"], groups=df["key"]))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(rest, rest["label"], groups=rest["key"]))
    train, val = rest.iloc[train_idx], rest.iloc[val_idx]

    return {
        name: part[["text", "label"]].reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")


def load_splits(split_dir: Path = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(split_dir / f"{name}.csv", encoding="utf-8", keep_default_na=False)
        for name in ("train", "val", "test")
    }


def main() -> None:
    df = load_raw(download())
    deduped, report = deduplicate(df)
    splits = split(deduped)
    write_splits(splits)

    print(report)
    for name, part in splits.items():
        counts = part["label"].value_counts().reindex(LABELS, fill_value=0)
        print(f"{name:5s} n={len(part):5d}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
