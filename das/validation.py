# dascore/validation.py
from __future__ import annotations
from typing import Any, Literal, Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

Task = Literal["classification", "regression"]


def _is_df(obj: Any) -> bool:
    return (pd is not None) and isinstance(obj, pd.DataFrame)


def _is_series(obj: Any) -> bool:
    return (pd is not None) and isinstance(obj, pd.Series)


def _to_numpy(X):
    if _is_df(X) or _is_series(X):
        return X.to_numpy()
    return np.asarray(X)


def _nrows(X) -> int:
    arr = _to_numpy(X)
    return int(arr.shape[0])


def _ncols(X) -> int:
    arr = _to_numpy(X)
    if arr.ndim < 2:
        raise ValueError("Feature matrix must be 2D (n_samples, n_features).")
    return int(arr.shape[1])


def _check_present(name: str, obj: Any) -> None:
    if obj is None:
        raise ValueError(f"{name} is required and cannot be None.")
    if _nrows(obj) == 0:
        raise ValueError(f"{name} must be non-empty.")


def _check_estimator_interface(estimator: Any) -> None:
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError(
            "Estimator must be an UNFITTED instance exposing fit(X, y) and predict(X)."
        )


def _check_columns_match(X_train: Any, X_syn: Any, X_test: Any) -> None:
    """If inputs are DataFrames, ensure columns (and order) match across all X."""
    if not (_is_df(X_train) and _is_df(X_syn) and _is_df(X_test)):
        return  # skip if not DataFrames
    cols_tr = list(X_train.columns)
    cols_sy = list(X_syn.columns)
    cols_te = list(X_test.columns)
    if not (cols_tr == cols_sy == cols_te):
        raise ValueError(
            "DataFrame columns must match (names and order) across X_train, X_syn, and X_test."
        )
    # Optional: dtype alignment (commented to keep it permissive)
    # if list(X_train.dtypes) != list(X_syn.dtypes) or list(X_train.dtypes) != list(X_test.dtypes):
    #     raise ValueError("DataFrame dtypes must match across X_train, X_syn, and X_test.")


def validate_inputs(
    *,
    X_train: Any,
    y_train: Any,
    X_syn: Any,
    y_syn: Any,
    X_test: Any,
    y_test: Optional[Any],
    task: Task,
    estimator: Any,
) -> None:
    """
    Runtime validation for DAS inputs.
    Keeps checks minimal and friendly; raises clear errors to aid users.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'.")

    _check_estimator_interface(estimator)

    # Presence / non-empty
    _check_present("X_train", X_train)
    _check_present("y_train", y_train)
    _check_present("X_syn", X_syn)
    _check_present("y_syn", y_syn)
    _check_present("X_test", X_test)

    if task == "regression":
        _check_present("y_test", y_test)
    # For classification, y_test is optional (agreement doesn’t use it)

    # Dimensionality
    n_features_train = _ncols(X_train)
    n_features_syn   = _ncols(X_syn)
    n_features_test  = _ncols(X_test)

    if not (n_features_train == n_features_syn == n_features_test):
        raise ValueError("All feature matrices must have the same number of columns.")

    # Length matches per split
    if _nrows(X_train) != _nrows(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if _nrows(X_syn) != _nrows(y_syn):
        raise ValueError("X_syn and y_syn must have the same number of rows.")
    if task == "regression" and _nrows(X_test) != _nrows(y_test):
        raise ValueError("X_test and y_test must have the same number of rows (regression).")

    # Optional: DataFrame column alignment
    _check_columns_match(X_train, X_syn, X_test)

    # (Optional) Basic NaN/inf checks — keep permissive by default.
    # Uncomment if you prefer strictness:
    # for name, arr in [("X_train", X_train), ("X_syn", X_syn), ("X_test", X_test),
    #                   ("y_train", y_train), ("y_syn", y_syn), ("y_test", y_test)]:
    #     if arr is None: 
    #         continue
    #     a = _to_numpy(arr)
    #     if not np.isfinite(a).all():
    #         raise ValueError(f"{name} contains NaN or inf. Clean or preprocess your data.")
