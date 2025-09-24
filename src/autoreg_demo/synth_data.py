
## src/autoreg_demo/synth_data.py
(Generates a tiny, labeled synthetic dataset. This stands in for your original preprocessing outputs.)
```python
import pandas as pd
from pathlib import Path
import random

OUT = Path(__file__).resolve().parents[2] / "data" / "synthetic_autoreg.csv"

def make_rows(n=100, seed=42):
    random.seed(seed)
    rows = []
    mech_terms = [
        "autoregulation", "autocatalysis", "autophosphorylation",
        "autoinhibition", "autoactivation"
    ]
    neutral_bio = [
        "Mitochondrial respiration increases ATP yield",
        "DNA repair maintains genome stability",
        "Protein folding assisted by chaperones",
        "Ion channels regulate membrane potential",
        "Histone modifications affect transcription"
    ]
    for i in range(n):
        if random.random() < 0.6:
            mech = random.choice(mech_terms)
            text = f"This study examines {mech} in kinase signaling pathways with evidence of {mech} in vitro."
            label = 1
        else:
            text = random.choice(neutral_bio)
            label = 0
        rows.append({"text": text, "label": label})
    return rows

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(make_rows())
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with shape {df.shape}")

if __name__ == "__main__":
    main()
