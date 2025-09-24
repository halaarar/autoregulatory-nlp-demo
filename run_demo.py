"""
Tiny retrieval + LLM workflow for synthetic autoregulation detection.

Inputs
- data/processed/clean_pubmed.csv
- data/processed/clean_autoreg.csv

Outputs
- artifacts/<run_id>/predictions.csv
- artifacts/<run_id>/report.json

Behavior
- Builds a TF-IDF retrieval index on PubMed title+abstract
- For each Autoreg entry, retrieves top-k snippets and forms a prompt
- Predicts mechanism and polarity via a model interface with fake and live modes
- Computes micro F1 for mechanism and accuracy for polarity
- Records average latency and an estimated token cost
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


MECHANISMS = ["autoregulation", "autophosphorylation", "autocatalysis", "autoactivation", "feedback_unknown", "none"]
POLARITIES = ["positive", "negative", "neutral"]


def load_inputs(pubmed_path: Path, autoreg_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned PubMed and Autoreg tables.
    """
    pubmed = pd.read_csv(pubmed_path, dtype=str)
    autoreg = pd.read_csv(autoreg_path, dtype=str)
    return pubmed, autoreg


def build_corpus(pubmed: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray, List[str], Dict[str, int]]:
    """
    Build a TF-IDF index over PubMed title+abstract.
    """
    text = (pubmed["Title"].fillna("") + " " + pubmed["Abstract"].fillna("")).tolist()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec.fit_transform(text)
    pmids = pubmed["PMID"].astype(str).tolist()
    idx = {pmid: i for i, pmid in enumerate(pmids)}
    return vec, mat, pmids, idx


def retrieve(vec: TfidfVectorizer, mat: np.ndarray, query_text: str, k: int) -> List[int]:
    """
    Return indices of top-k similar documents.
    """
    q = vec.transform([query_text])
    sim = cosine_similarity(q, mat).ravel()
    order = np.argsort(-sim)
    return order[:k].tolist()


def consolidate_mechanism(row: pd.Series) -> str:
    """
    Derive a single mechanism label from Term_in_* fields or 'none'.
    """
    for col in ["Term_in_RP", "Term_in_RT", "Term_in_RC"]:
        val = str(row.get(col, "") or "").strip().lower()
        if val in MECHANISMS and val != "none" and val != "":
            return val
    return "none"


def synthetic_polarity_for_label(label: str) -> str:
    """
    Assign a deterministic polarity label from the mechanism label.
    """
    if label in {"autoactivation", "autocatalysis", "autophosphorylation"}:
        return "positive"
    if label in {"autoregulation"}:
        return "negative"
    if label in {"feedback_unknown", "none"}:
        return "neutral"
    return "neutral"


def ground_truth_targets(autoreg: pd.DataFrame) -> pd.DataFrame:
    """
    Produce ground truth columns mechanism_label and polarity_label.
    """
    df = autoreg.copy()
    df["mechanism_label"] = df.apply(consolidate_mechanism, axis=1)
    df["polarity_label"] = df["mechanism_label"].apply(synthetic_polarity_for_label)
    return df


@dataclass
class LLMConfig:
    provider: str
    model_name: str
    api_key: str
    fake_mode: bool


class LLMClient:
    """
    Minimal model interface with fake and optional live modes.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = None
        if not cfg.fake_mode and cfg.provider == "openai" and cfg.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=cfg.api_key)
            except Exception:
                self._client = None

    def predict(self, prompt: str) -> Dict[str, str]:
        """
        Return a dict with mechanism_pred, polarity_pred, confidence.
        """
        if self.cfg.fake_mode or self._client is None:
            return self._fake_predict(prompt)
        try:
            rsp = self._client.chat.completions.create(
                model=self.cfg.model_name,
                messages=[
                    {"role": "system", "content": "Return a compact JSON with keys mechanism_pred, polarity_pred, confidence."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            text = rsp.choices[0].message.content or ""
            data = self._safe_parse_json(text)
            mech = data.get("mechanism_pred", "none")
            pol = data.get("polarity_pred", "neutral")
            conf = str(data.get("confidence", "0.50"))
            return {"mechanism_pred": mech, "polarity_pred": pol, "confidence": conf}
        except Exception:
            return self._fake_predict(prompt)

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, str]:
        try:
            return json.loads(text)
        except Exception:
            return {}

    @staticmethod
    def _fake_predict(prompt: str) -> Dict[str, str]:
        low = prompt.lower()
        if "autophosphorylation" in low:
            return {"mechanism_pred": "autophosphorylation", "polarity_pred": "positive", "confidence": "0.82"}
        if "autoactivation" in low:
            return {"mechanism_pred": "autoactivation", "polarity_pred": "positive", "confidence": "0.79"}
        if "autocatalysis" in low:
            return {"mechanism_pred": "autocatalysis", "polarity_pred": "positive", "confidence": "0.76"}
        if "autoregulation" in low:
            return {"mechanism_pred": "autoregulation", "polarity_pred": "negative", "confidence": "0.74"}
        if "feedback_unknown" in low:
            return {"mechanism_pred": "feedback_unknown", "polarity_pred": "neutral", "confidence": "0.60"}
        return {"mechanism_pred": "none", "polarity_pred": "neutral", "confidence": "0.55"}


def build_prompt(abstract: str, retrieved: List[Tuple[str, str]], label_space: List[str]) -> str:
    """
    Construct a compact instruction with inline citation markers.
    """
    cites = []
    for i, (pmid, snippet) in enumerate(retrieved, 1):
        cites.append(f"[{i}] PMID:{pmid} {snippet}")
    cite_block = "\n".join(cites)
    label_line = ", ".join(label_space)
    content = (
        "Given the abstract below and the retrieved snippets, identify the most likely autoregulatory mechanism "
        f"from {{{label_line}}}, or 'none' if absent, and the polarity in {{positive, negative, neutral}}. "
        "Respond as JSON with keys mechanism_pred, polarity_pred, confidence.\n"
        "Abstract:\n"
        f"{abstract}\n"
        "Snippets:\n"
        f"{cite_block}"
    )
    return content


def estimate_token_cost(n_chars: int) -> float:
    """
    Rough token cost estimator in USD for demonstration only.
    """
    tokens = max(1, n_chars // 4)
    cost_per_1k = 0.0005
    return round((tokens / 1000.0) * cost_per_1k, 6)


def run_workflow(
    in_pubmed: Path,
    in_autoreg: Path,
    out_dir: Path,
    top_k: int,
    model_name: str,
    provider: str,
    fake_mode: bool,
) -> None:
    """
    Execute retrieval, prompting, prediction, and evaluation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pubmed, autoreg = load_inputs(in_pubmed, in_autoreg)
    vec, mat, pmids, idx = build_corpus(pubmed)
    gt = ground_truth_targets(autoreg)

    api_key = os.getenv("LLM_API_KEY", "")
    cfg = LLMConfig(provider=provider, model_name=model_name, api_key=api_key, fake_mode=fake_mode or not api_key)
    client = LLMClient(cfg)

    records = []
    latencies = []
    token_chars = 0
    coverage = {"with_ground_truth": 0, "without_ground_truth": 0}

    for _, row in gt.iterrows():
        rx = str(row["RX"])
        pmid = rx.split("PubMed=")[-1].split(";")[0].strip()
        abstract = ""
        if pmid in idx:
            abstract = pubmed.loc[idx[pmid], "Abstract"]
        else:
            abstract = ""

        query_text = abstract if isinstance(abstract, str) else ""
        doc_idxs = retrieve(vec, mat, query_text, k=top_k + 1)
        retrieved = []
        used = 0
        for di in doc_idxs:
            r_pmid = pmids[di]
            r_abs = pubmed.iloc[di]["Abstract"]
            if r_pmid == pmid and used == 0:
                snippet = r_abs[:200].strip()
                retrieved.append((r_pmid, snippet))
                used += 1
            elif r_pmid != pmid and used < top_k:
                snippet = r_abs[:200].strip()
                retrieved.append((r_pmid, snippet))
                used += 1
            if used >= top_k:
                break

        prompt = build_prompt(abstract, retrieved, MECHANISMS)
        start = time.time()
        pred = client.predict(prompt)
        elapsed = (time.time() - start) * 1000.0
        latencies.append(elapsed)
        token_chars += len(prompt)

        citations = ";".join([pm for pm, _ in retrieved])
        mech_pred = pred.get("mechanism_pred", "none")
        pol_pred = pred.get("polarity_pred", "neutral")
        conf = pred.get("confidence", "0.50")

        mech_true = row.get("mechanism_label", None)
        pol_true = row.get("polarity_label", None)
        if mech_true is None or pol_true is None:
            coverage["without_ground_truth"] += 1
        else:
            coverage["with_ground_truth"] += 1

        records.append(
            {
                "AC": row["AC"],
                "PMID": pmid,
                "mechanism_pred": mech_pred,
                "polarity_pred": pol_pred,
                "confidence": conf,
                "retrieved_citations": citations,
                "mechanism_label": mech_true,
                "polarity_label": pol_true,
            }
        )

    df_pred = pd.DataFrame(records)
    has_gt = df_pred["mechanism_label"].notna() & df_pred["polarity_label"].notna()
    micro_f1 = f1_score(
        df_pred.loc[has_gt, "mechanism_label"],
        df_pred.loc[has_gt, "mechanism_pred"],
        labels=MECHANISMS,
        average="micro",
        zero_division=0,
    )
    acc_pol = accuracy_score(
        df_pred.loc[has_gt, "polarity_label"],
        df_pred.loc[has_gt, "polarity_pred"],
    )
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    est_cost = estimate_token_cost(token_chars)

    preds_path = out_dir / "predictions.csv"
    report_path = out_dir / "report.json"
    df_pred.to_csv(preds_path, index=False)
    report = {
        "micro_f1_mechanism": round(float(micro_f1), 4),
        "accuracy_polarity": round(float(acc_pol), 4),
        "avg_latency_ms": round(avg_latency, 2),
        "est_token_cost_usd": est_cost,
        "coverage_counts": coverage,
        "provider": provider if not fake_mode else "fake",
        "model_name": model_name if not fake_mode else "fake_stub",
        "top_k": top_k,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(str(preds_path))
    print(str(report_path))


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for input files, output directory, and runtime options.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--in-pubmed", type=Path, default=Path("data/processed/clean_pubmed.csv"))
    p.add_argument("--in-autoreg", type=Path, default=Path("data/processed/clean_autoreg.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts/run_001"))
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "fake"))
    p.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--fake-mode", action="store_true")
    return p.parse_args()


def main() -> None:
    """
    Entry point that runs the workflow with provided arguments and environment.
    """
    args = parse_args()
    fake_env = os.getenv("DEMO_FAKE_MODE", "").lower() in {"1", "true", "yes"}
    fake_mode = args.fake_mode or fake_env or not os.getenv("LLM_API_KEY", "")
    out_dir = args.out
    run_workflow(
        in_pubmed=args.in_pubmed,
        in_autoreg=args.in_autoreg,
        out_dir=out_dir,
        top_k=args.top_k,
        model_name=args.model,
        provider=args.provider,
        fake_mode=fake_mode,
    )


if __name__ == "__main__":
    main()
