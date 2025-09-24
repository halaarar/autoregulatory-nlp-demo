# Autoregulatory NLP Demo

A shareable, fully runnable demo that mirrors an autoregulatory-text pipeline using **synthetic data**.
- No proprietary data
- Offline runnable
- Includes tests

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate tiny synthetic dataset
python -m src.autoreg_demo.synth_data

# train and evaluate a tiny classifier
python -m src.autoreg_demo.model

# run tests
pytest
