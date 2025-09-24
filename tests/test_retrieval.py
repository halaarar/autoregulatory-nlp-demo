"""
Retrieval behavior on the TF-IDF index.
"""

import pandas as pd
import pytest

from run_demo import build_corpus, retrieve


def test_retrieval_returns_results():
    df_pub = pd.read_csv("data/processed/clean_pubmed.csv", dtype=str)
    vec, mat, pmids, idx = build_corpus(df_pub)
    query = (df_pub.iloc[0]["Title"] + " " + df_pub.iloc[0]["Abstract"]).strip()
    hits = retrieve(vec, mat, query, k=1)
    assert isinstance(hits, list)
    assert len(hits) == 1
    assert 0 <= hits[0] < len(df_pub)
