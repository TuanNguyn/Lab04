import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def count_selected_features(model, X):
    """Số feature thực sự đưa vào mô hình (hỗ trợ Pipeline có bước 'selector')."""
    if hasattr(model, "named_steps") and "selector" in model.named_steps:
        return int(model.named_steps["selector"].get_support().sum())
    return X.shape[1]


def evaluate_model(
    technique,
    name,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    n_splits,
    random_state,
):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_log_metrics = {"rmse": [], "mae": [], "r2": []}
    cv_original_metrics = {"rmse": [], "mae": [], "r2": []}
    fold_times = []

    for train_index, valid_index in cv.split(X_train):
        fold_model = clone(estimator)
        start = time.perf_counter()
        fold_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        predictions = fold_model.predict(X_train.iloc[valid_index])
        fold_times.append(time.perf_counter() - start)
        actual_log = y_train.iloc[valid_index].to_numpy()
        actual_original = np.expm1(actual_log)
        predicted_original = np.maximum(np.expm1(predictions), 0)

        cv_log_metrics["rmse"].append(
            mean_squared_error(actual_log, predictions) ** 0.5
        )
        cv_log_metrics["mae"].append(mean_absolute_error(actual_log, predictions))
        cv_log_metrics["r2"].append(r2_score(actual_log, predictions))
        cv_original_metrics["rmse"].append(
            mean_squared_error(actual_original, predicted_original) ** 0.5
        )
        cv_original_metrics["mae"].append(
            mean_absolute_error(actual_original, predicted_original)
        )
        cv_original_metrics["r2"].append(
            r2_score(actual_original, predicted_original)
        )

    fitted_model = clone(estimator)
    start = time.perf_counter()
    fitted_model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    test_predictions = fitted_model.predict(X_test)
    test_actual_log = y_test.to_numpy()
    test_actual_original = np.expm1(test_actual_log)
    test_predictions_original = np.maximum(np.expm1(test_predictions), 0)

    return {
        "technique": technique,
        "model": name,
        "cv_log_rmse": np.mean(cv_log_metrics["rmse"]),
        "cv_log_mae": np.mean(cv_log_metrics["mae"]),
        "cv_log_r2": np.mean(cv_log_metrics["r2"]),
        "cv_original_rmse": np.mean(cv_original_metrics["rmse"]),
        "cv_original_mae": np.mean(cv_original_metrics["mae"]),
        "cv_original_r2": np.mean(cv_original_metrics["r2"]),
        "test_log_rmse": mean_squared_error(test_actual_log, test_predictions) ** 0.5,
        "test_log_mae": mean_absolute_error(test_actual_log, test_predictions),
        "test_log_r2": r2_score(test_actual_log, test_predictions),
        "test_original_rmse": mean_squared_error(
            test_actual_original, test_predictions_original
        )
        ** 0.5,
        "test_original_mae": mean_absolute_error(
            test_actual_original, test_predictions_original
        ),
        "test_original_r2": r2_score(
            test_actual_original, test_predictions_original
        ),
        "n_features": count_selected_features(fitted_model, X_train),
        "cv_seconds": sum(fold_times),
        "train_seconds": train_seconds,
        "_fitted_model": fitted_model,
    }