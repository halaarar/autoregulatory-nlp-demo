"""
Preprocess synthetic PubMed and Autoregulatory raw tables.

Inputs
- data/raw/pubmed_synth.csv
- data/raw/autoreg_synth.csv

Outputs
- <out_dir>/clean_pubmed.csv
- <out_dir>/clean_autoreg.csv

Behavior
- Validates required columns and formats
- Trims whitespace, fills allowed nulls, drops invalid rows
- Verifies RX PubMed linkage to PubMed PMIDs
- Deduplicates within each table
- Prints a validation report
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


RE_PMID = re.compile(r"^\d{8}$")
RE_AC = re.compile(r"^[A-Z0-9]{6}$")
RE_RN = re.compile(r"^\[\d+\]$")
RE_RX = re.compile(r"PubMed\s*=\s*(\d{8})")

JOURNALS = {
    "Journal of Virology",
    "Virology",
    "Molecular Biology Reports",
    "Virus Research",
    "Microbiology Letters",
    "Infection and Immunity",
}

PUBMED_REQUIRED = ["PMID", "Title", "Abstract", "Journal", "Authors"]
AUTOREG_REQUIRED = [
    "AC", "OS", "RN", "RP", "RC", "RX", "RG", "RA", "RT", "RL",
    "Term_in_RP", "Term_in_RT", "Term_in_RC",
]


def _strip_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim leading and trailing whitespace from all string-like columns.
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df


def _collapse_spaces(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Replace multiple internal spaces with a single space for selected columns.
    """
    for col in cols:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
    return df


def _normalize_quotes(s: pd.Series) -> pd.Series:
    """
    Replace curly quotes with straight quotes.
    """
    return (
        s.astype(str)
        .str.replace("“", '"')
        .str.replace("”", '"')
        .str.replace("’", "'")
        .str.replace("‘", "'")
    )


def validate_and_clean_pubmed(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate schema and formats for PubMed data, then clean and deduplicate.
    Returns a cleaned DataFrame and a dict of counts dropped at each step.
    """
    report = {"input_rows": len(df)}

    missing_cols = [c for c in PUBMED_REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"PubMed missing required columns: {missing_cols}")

    df = df[PUBMED_REQUIRED].copy()
    df = _strip_all(df)
    df = _collapse_spaces(df, ["Title", "Abstract", "Authors"])

    before = len(df)
    df = df.dropna(subset=["PMID", "Title", "Abstract"])
    report["drop_pubmed_nulls"] = before - len(df)

    before = len(df)
    df = df[df["PMID"].astype(str).str.fullmatch(RE_PMID.pattern)]
    report["drop_pubmed_bad_pmid"] = before - len(df)

    before = len(df)
    df = df[df["Journal"].isin(JOURNALS)]
    report["drop_pubmed_bad_journal"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["drop_pubmed_dupes"] = before - len(df)

    df["PMID"] = df["PMID"].astype(str)

    if not df["PMID"].is_unique:
        raise ValueError("PubMed PMIDs are not unique after cleaning")

    report["output_rows"] = len(df)
    return df, report


def validate_and_clean_autoreg(df: pd.DataFrame, valid_pmids: pd.Series) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate schema and formats for Autoregulatory data, then clean, link, and deduplicate.
    Returns a cleaned DataFrame and a dict of counts dropped at each step.
    """
    report = {"input_rows": len(df)}

    missing_cols = [c for c in AUTOREG_REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Autoreg missing required columns: {missing_cols}")

    df = df[AUTOREG_REQUIRED].copy()
    df = _strip_all(df)
    df["RT"] = _normalize_quotes(df["RT"])
    df = _collapse_spaces(df, ["OS", "RP", "RC", "RX", "RG", "RA", "RT", "RL"])

    df["RC"] = df["RC"].fillna("")
    df["RG"] = df["RG"].fillna("")

    before = len(df)
    df = df.dropna(subset=["AC", "RX"])
    report["drop_autoreg_nulls"] = before - len(df)

    before = len(df)
    df = df[df["AC"].astype(str).str.fullmatch(RE_AC.pattern)]
    report["drop_autoreg_bad_ac"] = before - len(df)

    before = len(df)
    df = df[df["RN"].astype(str).str.fullmatch(RE_RN.pattern)]
    report["drop_autoreg_bad_rn"] = before - len(df)

    # Extract RX PMIDs and keep rows that have one
    rx_pmids = df["RX"].astype(str).str.extract(RE_RX, expand=False)
    before = len(df)
    df = df[rx_pmids.notna()].copy()
    report["drop_autoreg_bad_rx"] = before - len(df)

    # Normalize types and enforce linkage to PubMed
    df["RX_PMID"] = rx_pmids.loc[df.index].astype(str)
    valid_pmids = pd.Series(valid_pmids, dtype=str)
    before = len(df)
    df = df[df["RX_PMID"].isin(valid_pmids)]
    report["drop_autoreg_unlinked"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["drop_autoreg_dupes"] = before - len(df)

    if not df["AC"].is_unique:
        raise ValueError("Autoreg AC values are not unique after cleaning")

    report["output_rows"] = len(df)
    return df, report


def run(in_pubmed: Path, in_autoreg: Path, out_dir: Path) -> None:
    """
    Execute preprocessing and write cleaned outputs and a printed report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pubmed_raw = pd.read_csv(in_pubmed)
    autoreg_raw = pd.read_csv(in_autoreg)

    pubmed_clean, rep_pub = validate_and_clean_pubmed(pubmed_raw)
    autoreg_clean, rep_aut = validate_and_clean_autoreg(autoreg_raw, pubmed_clean["PMID"])

    pubmed_path = out_dir / "clean_pubmed.csv"
    autoreg_path = out_dir / "clean_autoreg.csv"
    pubmed_clean.to_csv(pubmed_path, index=False)
    autoreg_clean.to_csv(autoreg_path, index=False)

    print("Preprocessing report")
    print("PubMed")
    for k, v in rep_pub.items():
        print(f"- {k}: {v}")
    print("Autoregulatory")
    for k, v in rep_aut.items():
        print(f"- {k}: {v}")
    print(f"Wrote {pubmed_path}")
    print(f"Wrote {autoreg_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for input paths and output directory.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--in-pubmed", type=Path, required=True)
    p.add_argument("--in-autoreg", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    """
    Entry point that reads arguments and runs preprocessing.
    """
    args = parse_args()
    run(args["in_pubmed"] if isinstance(args, dict) else args.in_pubmed,
        args["in_autoreg"] if isinstance(args, dict) else args.in_autoreg,
        args["out_dir"] if isinstance(args, dict) else args.out_dir)


if __name__ == "__main__":
    main()
