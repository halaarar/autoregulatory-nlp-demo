from pathlib import Path
import pandas as pd
import subprocess
import sys

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_autoreg.csv"

def test_synth_generation_creates_file(tmp_path):
    # run generator
    res = subprocess.run(
        [sys.executable, "-m", "src.autoreg_demo.synth_data"],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    assert DATA.exists()
    df = pd.read_csv(DATA)
    assert {"text","label"}.issubset(df.columns)
    assert len(df) > 10
