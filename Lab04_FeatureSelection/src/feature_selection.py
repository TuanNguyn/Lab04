"""Xây Pipeline Feature Selection + mô hình cho Lab04.

Selector luôn nằm TRONG Pipeline (bước tên 'selector') nên khi cross-validation
được fit lại trên từng fold -> không rò rỉ (leakage) thông tin từ fold validation.
"""

from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42

TECHNIQUES = [
    "filter_kbest",       # Filter  : ANOVA F-test (f_regression)
    "filter_mi",          # Filter  : Mutual Information
    "wrapper_rfe",        # Wrapper : Recursive Feature Elimination
    "embedded_lasso",     # Embedded: hệ số Lasso (L1)
    "embedded_rf",        # Embedded: feature importance của RandomForest
]


def make_models():
    """Siêu tham số GIỮ NGUYÊN như baseline (02_baseline_modeling)."""
    return {
        "SVR": SVR(C=10.0, epsilon=0.01, kernel="rbf"),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", eval_metric="rmse",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }


def _rfe_estimator(model_name):
    """Estimator dùng để xếp hạng feature trong RFE (SVR rbf không có coef_)."""
    if model_name == "SVR":
        return Ridge(alpha=10.0)
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=50, max_features=0.3,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    return XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=1,
    )


def make_selector(technique, model_name, k):
    if technique == "filter_kbest":
        return SelectKBest(score_func=f_regression, k=k)
    if technique == "filter_mi":
        return SelectKBest(
            score_func=partial(mutual_info_regression, random_state=RANDOM_STATE), k=k
        )
    if technique == "wrapper_rfe":
        return RFE(_rfe_estimator(model_name), n_features_to_select=k, step=20)
    if technique == "embedded_lasso":
        # Số feature do Lasso tự quyết định (hệ số khác 0), không cố định = k
        return SelectFromModel(
            LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=50000), threshold=1e-5
        )
    if technique == "embedded_rf":
        return SelectFromModel(
            RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            max_features=k, threshold=-float("inf"),
        )
    raise ValueError(technique)


def make_pipeline(technique, model_name, estimator, k=50):
    return Pipeline([
        ("selector", make_selector(technique, model_name, k)),
        ("model", estimator),
    ])
