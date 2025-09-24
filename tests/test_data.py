"""
Data integrity tests for cleaned inputs.
"""

import re
from pathlib import Path

import pandas as pd
import pytest


def test_clean_files_exist_and_nonempty():
    pubmed_path = Path("data/processed/clean_pubmed.csv")
    autoreg_path = Path("data/processed/clean_autoreg.csv")
    assert pubmed_path.exists()
    assert autoreg_path.exists()
    df_pub = pd.read_csv(pubmed_path, dtype=str)
    df_aut = pd.read_csv(autoreg_path, dtype=str)
    assert len(df_pub) > 0
    assert len(df_aut) > 0


def test_rx_pubmed_linkage():
    df_pub = pd.read_csv("data/processed/clean_pubmed.csv", dtype=str)
    df_aut = pd.read_csv("data/processed/clean_autoreg.csv", dtype=str)

    pub_pmids = set(df_pub["PMID"].astype(str))

    if "RX_PMID" in df_aut.columns:
        rx_pmids = df_aut["RX_PMID"].astype(str)
    else:
        rx_pmids = df_aut["RX"].astype(str).str.extract(r"PubMed\s*=\s*(\d{8})", expand=False)

    assert rx_pmids.notna().all()
    assert set(rx_pmids).issubset(pub_pmids)


def test_required_columns_present():
    df_pub = pd.read_csv("data/processed/clean_pubmed.csv", dtype=str)
    df_aut = pd.read_csv("data/processed/clean_autoreg.csv", dtype=str)

    pub_required = {"PMID", "Title", "Abstract", "Journal", "Authors"}
    aut_required = {
        "AC", "OS", "RN", "RP", "RC", "RX", "RG", "RA", "RT", "RL",
        "Term_in_RP", "Term_in_RT", "Term_in_RC"
    }

    assert pub_required.issubset(set(df_pub.columns))
    assert aut_required.issubset(set(df_aut.columns))
