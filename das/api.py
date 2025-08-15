# dascore/api.py
from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from copy import deepcopy
import numpy as np
import pandas as pd

from .metrics import das_classification, das_regression
from .validation import validate_inputs

Task = Literal["classification", "regression"]

def _to_numpy(X):
    """Convert pandas DataFrame/Series to numpy; otherwise np.asarray."""
    if pd is not None and isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)

def evaluate_das(
    X_train,
    y_train,
    X_syn,
    y_syn,
    X_test,
    y_test: Optional[Any] = None,
    *,
    task: Task = "classification",
    estimator: Any,
    auto_preprocess: bool = False,  # reserved for future use; identity for now
) -> Dict[str, Any]:
    """
    Compute the Data Accuracy Score (DAS) between ORIGINAL and SYNTHETIC training data.

    - Classification DAS: % agreement between predictions from
      (estimator trained on original) vs (estimator trained on synthetic), both evaluated on X_test.
    - Regression DAS: ((R2_original - R2_synthetic) / max(eps, R2_original)) * 100,
      where scores are on (X_test, y_test).

    Parameters
    ----------
    X_train, y_train : original training features/labels
    X_syn,   y_syn   : synthetic training features/labels (schema-aligned with X_train)
    X_test,  y_test  : original test features and (for regression) labels
    task             : "classification" or "regression"
    estimator        : UNFITTED model instance exposing .fit(X, y) and .predict(X)
    auto_preprocess  : placeholder flag (no-op). Hook up a pipeline later without changing the API.

    Returns
    -------
    dict with keys:
        - task
        - estimator (class name)
        - das (float)
        - r2_original, r2_synthetic  (only for regression)
    """
    # All checks (including estimator API) happen in validation.py
    validate_inputs(
        X_train=X_train, y_train=y_train,
        X_syn=X_syn, y_syn=y_syn,
        X_test=X_test, y_test=y_test,
        task=task, estimator=estimator,
    )

    # (Optional) preprocessing hook — identity for now
    Xtr = _to_numpy(X_train)
    Xsy = _to_numpy(X_syn)
    Xte = _to_numpy(X_test)

    ytr = _to_numpy(y_train)
    ysy = _to_numpy(y_syn)
    yte = None if y_test is None else _to_numpy(y_test)

    # Two independent copies: one per training arm
    est_orig = deepcopy(estimator)
    est_syn  = deepcopy(estimator)

    if task == "classification":
        est_orig.fit(Xtr, ytr)
        pred_o = est_orig.predict(Xte)

        est_syn.fit(Xsy, ysy)
        pred_s = est_syn.predict(Xte)

        das = float(das_classification(pred_o, pred_s))
        return {
            "task": task,
            "estimator": est_orig.__class__.__name__,
            "das": das,
        }

    # Regression
    est_orig.fit(Xtr, ytr)
    ypo = est_orig.predict(Xte)

    est_syn.fit(Xsy, ysy)
    yps = est_syn.predict(Xte)

    das, r2o, r2s = das_regression(yte, ypo, yps)
    return {
        "task": task,
        "estimator": est_orig.__class__.__name__,
        "das": float(das),
        "r2_original": float(r2o),
        "r2_synthetic": float(r2s),
    }
