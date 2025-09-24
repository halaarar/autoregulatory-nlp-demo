# Autoregulation NLP Demo

This repository is a safe, self-contained demonstration of my capstone project on detecting protein autoregulatory mechanisms in biomedical text. The original capstone used partner data and private model assets that I cannot share. To respect confidentiality, this demo recreates the workflow on synthetic data with a lightweight retrieval plus LLM pipeline that is fully runnable, testable, and easy to review.

## What this is
- A minimal end to end workflow that mirrors the capstone structure on non-sensitive, synthetic inputs.
- The goal is to detect a single autoregulatory mechanism label and a polarity label from PubMed-style abstracts.
- Retrieval uses local TF-IDF over the synthetic abstracts. The model step supports an offline fake mode so it runs without keys or network access.
- Outputs include predictions with inline citation identifiers and a small report with quality and runtime metrics.
- This demo runs offline by default with `--fake-mode`. No API keys are required.

## What this is not
- This is not the original capstone repository. It does not include any partner data, internal prompts, private checkpoints, or deployment code.
- Results are illustrative. Metrics reflect synthetic labels and a small local retrieval index, not production performance.

## Why synthetic data
The capstone was completed with a university partner and included domain material and annotations that are not public. This demo preserves the original schema shape and linkage patterns while generating entirely synthetic rows. It allows reviewers to run the full workflow and tests without exposing confidential information.

## How this maps to the Impact Case Study (my capstone project)
- Problem: same task definition on abstracts, focused on mechanism and polarity.
- Data flow: raw-like tables to preprocessing and validation to retrieval to prompt to predictions to evaluation.
- Modeling: the capstone fine-tuned PubMedBERT with multi-task loss. The demo uses a tiny retrieval plus prompting pipeline with a fake mode so it is simple and runs anywhere. This aligns with the role requirement for a small LLM workflow with citations and a tiny eval.
- Evaluation: micro F1 for mechanism, accuracy for polarity, plus latency and a simple cost estimate.

---

## Demo in 60 Seconds
1) Goal: detect an autoregulatory mechanism and polarity from synthetic abstracts using tiny retrieval plus LLM.
2) Inputs: `data/processed/clean_pubmed.csv`, `data/processed/clean_autoreg.csv`.
3) Command: `python run_demo.py --fake-mode`
4) Outputs: `artifacts/run_001/predictions.csv`, `artifacts/run_001/report.json`
5) Eval: micro F1 (mechanism), accuracy (polarity), average latency, and a simple token cost estimate.

---

## Setup
Use Python 3.11 if possible.

Clone the repository 
```bash
git clone https://github.com/halaarar/autoregulatory-nlp-demo.git
```

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for tests
```

## Create Synthetic Data
```bash
python 00_make_synthetic_raw.py
python 01_preprocess.py --in-pubmed data/raw/pubmed_synth.csv --in-autoreg data/raw/autoreg_synth.csv --out-dir data/processed
```

## Run the demo 
```bash
python run_demo.py --fake-mode
```

## Run tests
```bash
pytest -q
```

## Repository Structure

```
.
├── 00_make_synthetic_raw.py        # creates data/raw/*.csv with synthetic rows
├── 01_preprocess.py                # validates, cleans, links, writes data/processed/*.csv
├── run_demo.py                     # retrieval + prompting + predictions + report
├── data/
│   ├── raw/                        # synthetic inputs
│   └── processed/                  # cleaned tables
├── artifacts/                      # predictions and report.json
├── tests/                          # small test suite for data, retrieval, and evals
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Outputs 

- artifacts/run_001/predictions.csv
Columns: AC, PMID, mechanism_pred, polarity_pred, confidence retrieved_citations, mechanism_label, polarity_label

- artifacts/run_001/report.json
Keys: micro_f1_mechanism, accuracy_polarity, avg_latency_ms, est_token_cost_usd, coverage_counts, provider, model_name, top_k

## Data Dictionary
**Inputs**
clean_pubmed.csv
- PMID string, 8 digits
- Title string
- Abstract string
- Journal categorical from a small whitelist
- Authors comma separated Surname I. entries

clean_autoreg.csv
- AC string, 6 char accession
- RX string containing PubMed=<PMID> reference
- RT, RL, RN, RP, RC, RG, RA, OS descriptive strings
- Term_in_RP, Term_in_RT, Term_in_RC optional mechanism keywords or empty

**Outputs**

predictions.csv
- AC, PMID
- mechanism_pred in {autoregulation, autophosphorylation, autocatalysis, autoactivation, feedback_unknown, none}
- polarity_pred in {positive, negative, neutral}
- confidence string score
- retrieved_citations semicolon separated PMIDs used for context
- mechanism_label, polarity_label synthetic ground truth for scoring

**report.json**
- micro_f1_mechanism, accuracy_polarity
- avg_latency_ms, est_token_cost_usd
- coverage_counts, provider, model_name, top_k

## Model behavior at a glance

- Builds a TF-IDF index over Title + Abstract and retrieves top snippets for each target record.
- Forms a compact prompt with inline citation markers that list the retrieved PMIDs.
- In offline mode, returns deterministic predictions from a small rules stub so the demo runs without network access.
- Computes micro F1 for mechanism and accuracy for polarity using synthetic labels.