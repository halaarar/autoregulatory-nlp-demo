"""
Generate fully synthetic raw datasets for the demo.

Outputs
- data/raw/pubmed_synth.csv
- data/raw/autoreg_synth.csv

Design
- Deterministic generation with a fixed seed
- Simple value vocabularies to avoid external dependencies
- Light schema validation to ensure join integrity via PubMed IDs
"""

from __future__ import annotations

import csv
import random
import string
from pathlib import Path
from typing import List

import pandas as pd


SEED = 4242
random.seed(SEED)

JOURNALS: List[str] = [
    "Journal of Virology",
    "Virology",
    "Molecular Biology Reports",
    "Virus Research",
    "Microbiology Letters",
    "Infection and Immunity",
]

ORGANISMS: List[str] = [
    "Frog virus 3 (isolate Delta) (FV-3)",
    "Invertebrate iridescent virus 3 (Mosquito iridescent virus) (IIV-3)",
    "Invertebrate iridescent virus 6 (Chilo iridescent virus) (IIV-6)",
    "Marine alga dsDNA virus 1",
    "Baculovirus X isolate Northwood",
    "Synthetic orthomyxo-like virus A",
]

RP_PHRASES: List[str] = [
    "NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].",
    "GENOME REANNOTATION.",
    "TRANSCRIPTIONAL PROFILING.",
    "PROTEIN FUNCTION CHARACTERIZATION.",
]

NOTES_RC: List[str] = [
    "",
    "",
    "Contains tandem repeats in the late region.",
    "Putative membrane association motif identified.",
    "",
]

OPTIONAL_TERMS: List[str] = [
    "autoregulation",
    "autophosphorylation",
    "autocatalysis",
    "autoactivation",
    "feedback_unknown",
]

SURNAMES: List[str] = [
    "Tan", "Barkman", "Chinchar", "Eaton", "Metcalf", "Penny", "Whitley",
    "Yu", "Sample", "Delhon", "Tulman", "Afonso", "Jakob", "Mueller",
    "Bahr", "Darai", "Cackett", "Matelska", "Sykora", "Pain", "Renauld",
    "Berriman", "Iams", "Young", "Nene", "Desai", "Webster"
]

INITIALS: List[str] = list(string.ascii_uppercase)


def _project_paths() -> tuple[Path, Path]:
    """
    Return repository root and data/raw paths, creating data/raw if needed.
    """
    here = Path(__file__).resolve()
    project_root = here.parent
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)
    return project_root, data_raw


def _make_pmid(i: int) -> str:
    """
    Create an 8 digit synthetic PubMed identifier string.
    """
    return f"{90000000 + i:08d}"


def _make_title() -> str:
    """
    Create a short synthetic PubMed-style title.
    """
    templates = [
        "Comparative genomic analysis of {org}",
        "Transcriptome profiling of {org}",
        "Functional annotation of regulatory proteins in {org}",
        "Genome architecture and coding potential of {org}",
        "Host interaction factors shaping {org} infection",
    ]
    base = random.choice(templates)
    org = random.choice(ORGANISMS).split(" (")[0]
    return base.format(org=org)


def _make_abstract() -> str:
    """
    Create a multi sentence abstract with an optional mechanism keyword.
    """
    base = [
        "We investigated a double stranded DNA virus using comparative analyses of coding sequences.",
        "The study assessed regulatory motifs and conserved domains across assembled contigs.",
        "Results suggest context dependent control of gene expression in infected cells.",
        "We evaluated sequence features using alignment, clustering, and topic extraction.",
        "Findings highlight constraints relevant to classification, ranking, and summarization.",
    ]
    text = " ".join(base)
    if random.random() < 0.6:
        text += " This work discusses {} and related self regulation phenomena.".format(
            random.choice(OPTIONAL_TERMS)
        )
    return text


def _make_authors(n_min: int = 2, n_max: int = 6) -> str:
    """
    Create a comma separated author string in Surname I. format.
    """
    n = random.randint(n_min, n_max)
    chosen = random.sample(SURNAMES, k=n)
    parts: List[str] = []
    for s in chosen:
        k = random.randint(1, 2)
        initials = ".".join(random.sample(INITIALS, k=k)) + "."
        parts.append(f"{s} {initials}")
    return ", ".join(parts)


def _make_doi(i: int) -> str:
    """
    Create a deterministic fake DOI string.
    """
    return f"10.1234/synth.{i:05d}"


def _make_rl() -> str:
    """
    Create a journal citation line of the form 'Name vol:start-end(year).'
    """
    journal = random.choice(JOURNALS)
    vol = random.randint(100, 399)
    start = random.randint(1, 800)
    end = start + random.randint(5, 50)
    year = random.randint(2000, 2022)
    return f"{journal} {vol}:{start}-{end}({year})."


def _make_ac() -> str:
    """
    Create a 6 character uppercase alphanumeric accession string.
    """
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(6))


def generate_pubmed(n_rows: int = 50) -> pd.DataFrame:
    """
    Generate a synthetic PubMed-like table with fixed schema.

    Columns
    - PMID: 8 digit string
    - Title: short sentence
    - Abstract: multi sentence paragraph
    - Journal: selected from a whitelist
    - Authors: 'Surname I.' entries separated by commas
    """
    rows = []
    for i in range(1, n_rows + 1):
        rows.append(
            {
                "PMID": _make_pmid(i),
                "Title": _make_title(),
                "Abstract": _make_abstract(),
                "Journal": random.choice(JOURNALS),
                "Authors": _make_authors(),
            }
        )
    df = pd.DataFrame(rows)
    assert df["PMID"].is_unique
    assert df["PMID"].str.fullmatch(r"\d{8}").all()
    assert df["Title"].str.len().gt(0).all()
    assert df["Abstract"].str.len().between(100, 2000).all()
    assert df["Journal"].isin(JOURNALS).all()
    return df


def generate_autoreg(pubmed_df: pd.DataFrame, n_rows: int = 80) -> pd.DataFrame:
    """
    Generate a synthetic autoregulatory table that references PubMed PMIDs.

    Columns
    - AC: 6 character uppercase alphanumeric accession
    - OS: organism string
    - RN: bracketed integer citation count
    - RP: short uppercase description phrase
    - RC: optional notes
    - RX: 'PubMed=<PMID>; DOI=<fake-doi>;' referencing the synthetic PubMed table
    - RG: optional group string
    - RA: author list mirroring PubMed style
    - RT: quoted short title derived from the linked PubMed record
    - RL: journal citation line
    - Term_in_RP, Term_in_RT, Term_in_RC: optional mechanism keywords or empty
    """
    pmids = pubmed_df["PMID"].tolist()
    rows = []
    for i in range(1, n_rows + 1):
        pmid = random.choice(pmids)
        pm_title = pubmed_df.loc[pubmed_df["PMID"] == pmid, "Title"].values[0]
        rows.append(
            {
                "AC": _make_ac(),
                "OS": random.choice(ORGANISMS),
                "RN": f"[{random.randint(1,3)}]",
                "RP": random.choice(RP_PHRASES),
                "RC": random.choice(NOTES_RC),
                "RX": f"PubMed={pmid}; DOI={_make_doi(i)};",
                "RG": "" if random.random() < 0.6 else "Consortium Study Group",
                "RA": _make_authors(),
                "RT": f"\"{pm_title[:70]}...\"",
                "RL": _make_rl(),
                "Term_in_RP": random.choice(OPTIONAL_TERMS + [""]),
                "Term_in_RT": random.choice(OPTIONAL_TERMS + [""]),
                "Term_in_RC": random.choice(OPTIONAL_TERMS + [""]),
            }
        )
    df = pd.DataFrame(rows)
    assert df["AC"].is_unique
    assert df["AC"].str.fullmatch(r"[A-Z0-9]{6}").all()
    assert df["RN"].str.fullmatch(r"\[\d+\]").all()
    assert df["RX"].str.contains(r"PubMed=\d{8}; DOI=10\.1234/synth\.\d{5};").all()
    rx_pmids = df["RX"].str.extract(r"PubMed=(\d{8})")[0]
    assert set(rx_pmids).issubset(set(pmids))
    return df


def main() -> None:
    """
    Generate and save both synthetic raw datasets to CSV under data/raw.
    """
    _, data_raw = _project_paths()

    pubmed_df = generate_pubmed(n_rows=50)
    pubmed_path = data_raw / "pubmed_synth.csv"
    pubmed_df.to_csv(pubmed_path, index=False, quoting=csv.QUOTE_MINIMAL)

    autoreg_df = generate_autoreg(pubmed_df, n_rows=80)
    autoreg_path = data_raw / "autoreg_synth.csv"
    autoreg_df.to_csv(autoreg_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {pubmed_path}")
    print(f"Wrote {autoreg_path}")


if __name__ == "__main__":
    main()
