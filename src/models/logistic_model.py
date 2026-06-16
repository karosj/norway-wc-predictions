"""
Logistic regression model + time-based evaluation.

Multinomial logistic on Elo gap + rolling-form diffs + context features,
paired with the Poisson model for scoreline distributions. Together these
are the two-model core of the prediction pipeline.

Hyperparameters (from walk-forward CV):
    C=1.0, pure L2 via saga — C in [1, 100] all ~equivalent; 1.0 keeps mild
    regularisation without sacrificing CV log-loss. l1_ratio=0 = pure L2.
Time-decay half-life = 1095 days (~3 yr) — CV-neutral but encodes recency
    preference. Aggressive decay (<365 d) hurts logistic on small samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.config import (
    MODEL_FEATURES, RESULT_CLASSES,
    TIME_DECAY_HALF_LIFE_DAYS, TRAIN_TEST_SPLIT_DATE,
)
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def make_logistic() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, l1_ratio=0.0, solver="saga", max_iter=5000)),
    ])


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #

def proba_frame(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """predict_proba -> DataFrame with fixed win/draw/loss column order."""
    proba = model.predict_proba(X)
    cls = list(model.named_steps["clf"].classes_)
    return pd.DataFrame(
        {c: proba[:, cls.index(c)] if c in cls else 0.0 for c in RESULT_CLASSES},
        index=X.index,
    )


# sklearn log_loss expects y_prob columns in lexicographic sorted order.
_SORTED_CLASSES = sorted(RESULT_CLASSES)


def safe_log_loss(y_true, proba_df: pd.DataFrame) -> float:
    """log_loss with columns aligned to sorted labels (avoids silent misorder)."""
    return log_loss(y_true, proba_df[_SORTED_CLASSES].values, labels=_SORTED_CLASSES)


# --------------------------------------------------------------------------- #
# Time-decay sample weights (tidsavskrivning)
# --------------------------------------------------------------------------- #

def time_decay_weights(dates: pd.Series, reference: pd.Timestamp,
                       half_life_days: float | None) -> np.ndarray | None:
    """Exponential recency weight: 0.5 ** (age_days / half_life), mean-normalised.

    Mean-1 normalisation keeps total data-loss magnitude consistent with the
    unweighted fit so L2 C stays balanced. Returns None to disable weighting.
    """
    if half_life_days is None:
        return None
    age = (reference - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    w = 0.5 ** (age / half_life_days)
    return w * (len(w) / w.sum())


def _fit_kwargs(weights: np.ndarray | None) -> dict:
    return {"clf__sample_weight": weights} if weights is not None else {}


def _labelled(ds: pd.DataFrame) -> pd.DataFrame:
    return ds[ds["target_result"].notna()].copy()


def _assert_split_isolation(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Guard against a match_id spanning the train/test boundary.

    Both perspective rows of a match share a match_id; a date-based split keeps
    them together. This assertion makes that explicit and will fail loudly if a
    future schema change ever lets a match span the boundary.
    """
    overlap = set(train["match_id"]) & set(test["match_id"])
    if overlap:
        raise ValueError(
            f"Train/test leakage: {len(overlap)} match_id(s) span the split "
            f"boundary (e.g. {sorted(overlap)[:3]}). Both perspectives of a "
            "match must stay on the same side of the split.")


# --------------------------------------------------------------------------- #
# Time-based evaluation
# --------------------------------------------------------------------------- #

def evaluate_time_split(ds: pd.DataFrame,
                        train_override: pd.DataFrame | None = None,
                        half_life_days: float | None = TIME_DECAY_HALF_LIFE_DAYS,
                        ) -> dict:
    """Train on pre-2025 matches, test on 2025+; report accuracy + log-loss.

    train_override: use this DataFrame for training instead of ds (e.g. the
    all-team dataset for broader coverage); test set is always from ds.
    """
    ds = _labelled(ds)
    train_default = ds[ds["date"] < TRAIN_TEST_SPLIT_DATE]
    test = ds[ds["date"] >= TRAIN_TEST_SPLIT_DATE]
    _assert_split_isolation(train_default, test)

    if train_override is not None:
        train = _labelled(train_override)
        train = train[train["date"] < TRAIN_TEST_SPLIT_DATE]
    else:
        train = train_default

    log.info("time split: %d train / %d test | half-life=%s d",
             len(train), len(test), half_life_days)
    _reference_baselines(train_default, test)

    weights = time_decay_weights(train["date"], train["date"].max(), half_life_days)
    model = make_logistic()
    model.fit(train[MODEL_FEATURES], train["target_result"], **_fit_kwargs(weights))
    proba = proba_frame(model, test[MODEL_FEATURES])
    pred = proba[RESULT_CLASSES].idxmax(axis=1)
    acc = accuracy_score(test["target_result"], pred)
    ll = safe_log_loss(test["target_result"], proba)

    print("\n" + "=" * 70)
    print(f"MODEL: logistic   ({len(MODEL_FEATURES)} features)")
    print("=" * 70)
    print(f"  accuracy: {acc:.3f}   log-loss: {ll:.3f}")
    print("  classification report:")
    print(classification_report(test["target_result"], pred,
                                labels=RESULT_CLASSES, zero_division=0))
    cm = confusion_matrix(test["target_result"], pred, labels=RESULT_CLASSES)
    print("  confusion matrix (rows=true, cols=pred) "
          f"[{', '.join(RESULT_CLASSES)}]:")
    print("   " + str(cm).replace("\n", "\n   "))
    _calibration_report(train, test, weights)

    return {"accuracy": acc, "log_loss": ll, "n_test": len(test)}


def _reference_baselines(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Naive baselines so model metrics can be read in context."""
    y = test["target_result"]
    prior = train["target_result"].value_counts(normalize=True)
    prior_df = pd.DataFrame([{c: prior.get(c, 0.0) for c in RESULT_CLASSES}] * len(test))
    uniform_df = pd.DataFrame([{c: 1 / 3 for c in RESULT_CLASSES}] * len(test))
    majority = train["target_result"].value_counts().idxmax()

    print("\n" + "=" * 70)
    print("REFERENCE BASELINES  (test set)")
    print("=" * 70)
    print(f"  always-'{majority}'   accuracy {(y == majority).mean():.3f}")
    print(f"  class-prior      accuracy {(y == prior.idxmax()).mean():.3f}"
          f"   log-loss {safe_log_loss(y, prior_df):.3f}")
    print(f"  uniform (1/3)                     "
          f"   log-loss {safe_log_loss(y, uniform_df):.3f}")


def _calibration_report(train: pd.DataFrame, test: pd.DataFrame,
                        weights: np.ndarray | None = None) -> None:
    """P(win) reliability in probability bins (logistic, test set)."""
    model = make_logistic()
    model.fit(train[MODEL_FEATURES], train["target_result"], **_fit_kwargs(weights))
    p_win = proba_frame(model, test[MODEL_FEATURES])["win"]
    actual_win = (test["target_result"] == "win").astype(int)
    bins = pd.cut(p_win, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    table = pd.DataFrame({"p_win": p_win, "won": actual_win}).groupby(bins, observed=True)
    print("\n" + "=" * 70)
    print("CALIBRATION  P(win): predicted vs observed (logistic, test set)")
    print("=" * 70)
    print(f"  {'bin':<14}{'n':>5}{'mean_pred':>12}{'observed':>12}")
    for b, g in table:
        if len(g):
            print(f"  {str(b):<14}{len(g):>5}{g['p_win'].mean():>12.2f}"
                  f"{g['won'].mean():>12.2f}")


# --------------------------------------------------------------------------- #
# Final fit on all data (for prediction)
# --------------------------------------------------------------------------- #

def fit_final(ds: pd.DataFrame,
              train_override: pd.DataFrame | None = None,
              half_life_days: float | None = TIME_DECAY_HALF_LIFE_DAYS,
              ) -> dict:
    """Fit logistic on all labelled matches; return {"logistic": (model, feats)}.

    train_override: optional DataFrame to train on instead of ds (e.g. the
    all-team dataset for broader coverage).
    """
    train = _labelled(train_override if train_override is not None else ds)
    weights = time_decay_weights(train["date"], train["date"].max(), half_life_days)
    model = make_logistic()
    model.fit(train[MODEL_FEATURES], train["target_result"], **_fit_kwargs(weights))
    log.info("fitted logistic on %d rows (half-life=%s d)", len(train), half_life_days)
    return {"logistic": (model, MODEL_FEATURES)}
