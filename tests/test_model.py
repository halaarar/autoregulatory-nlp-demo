import sys
import subprocess

def test_model_runs_end_to_end():
    # Ensure data exists
    subprocess.run([sys.executable, "-m", "src.autoreg_demo.synth_data"], check=True)
    # Train + eval
    res = subprocess.run(
        [sys.executable, "-m", "src.autoreg_demo.model"],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    # basic sanity checks in output
    assert "Precision:" in res.stdout
    assert "Recall:" in res.stdout
    assert "F1:" in res.stdout
