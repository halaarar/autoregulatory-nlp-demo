"""
Evaluator metrics run without error and score perfectly on a toy match.
"""

import pandas as pd
import pytest
from sklearn.metrics import f1_score, accuracy_score

from run_demo import ground_truth_targets, MECHANISMS


@pytest.mark.skipif(
    len(pd.read_csv("data/processed/clean_autoreg.csv")) == 0,
    reason="No autoregulatory rows available",
)
def test_metrics_compute_and_match_when_predictions_equal_labels():
    df_aut = pd.read_csv("data/processed/clean_autoreg.csv", dtype=str)
    gt = ground_truth_targets(df_aut)

    y_true_mech = gt["mechanism_label"]
    y_true_pol = gt["polarity_label"]

    y_pred_mech = y_true_mech.copy()
    y_pred_pol = y_true_pol.copy()

    micro_f1 = f1_score(y_true_mech, y_pred_mech, labels=MECHANISMS, average="micro", zero_division=0)
    acc = accuracy_score(y_true_pol, y_pred_pol)

    assert micro_f1 == 1.0
    assert acc == 1.0
