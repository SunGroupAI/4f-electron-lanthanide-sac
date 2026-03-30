import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-out model selection and missing-value imputation for multi-target tabular data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output CSV file for the imputed dataset."
    )
    return parser.parse_args()


def create_model(model_name):
    if model_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
    elif model_name == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=10, kernel="rbf"))
        ])
    elif model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            ))
        ])
    elif model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_model(model, X, y, loo):
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_true_all.append(y[test_idx][0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)

    return mse, r2, mae


def main():
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file

    data = pd.read_csv(input_file)

    identifier_columns = ["name", "element"]
    target_columns = [
        "h", "c", "n", "f", "cl", "s", "co2", "cho", "co", "cooh",
        "n2", "n2h", "no", "noh", "oh", "o2", "h2o"
    ]
    feature_columns = [col for col in data.columns if col not in identifier_columns + target_columns]
    X_all = data[feature_columns].values

    loo = LeaveOneOut()

    all_model_results = []
    best_model_results = []

    candidate_models = [
        "ridge",
        "random_forest",
        "xgboost",
        "svr",
        "mlp",
        "hist_gradient_boosting"
    ]

    for target in target_columns:
        print(f"\\nProcessing target: {target}")

        y = data[target].values
        missing_mask = np.isnan(y)
        known_mask = ~missing_mask

        X_known = X_all[known_mask]
        y_known = y[known_mask]
        X_missing = X_all[missing_mask]

        if len(y_known) < 2:
            print(f"Skipping {target}: insufficient samples.")
            continue

        best_r2 = -np.inf
        best_model_name = None
        best_model = None
        best_mse = None
        best_mae = None

        for model_name in candidate_models:
            model = create_model(model_name)
            try:
                mse, r2, mae = evaluate_model(model, X_known, y_known, loo)
                print(f"  Model {model_name} - R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

                all_model_results.append({
                    "target": target,
                    "model": model_name,
                    "r2": r2,
                    "mse": mse,
                    "mae": mae
                })

                if not np.isnan(r2) and r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_model = model
                    best_mse = mse
                    best_mae = mae

            except Exception as e:
                print(f"Model {model_name} failed on target {target}: {e}")
                all_model_results.append({
                    "target": target,
                    "model": model_name,
                    "r2": np.nan,
                    "mse": np.nan,
                    "mae": np.nan
                })

        if best_model is None:
            print(f"All models failed on target {target}. Skipping imputation.")
            continue

        print(f"Selected model for {target}: {best_model_name}, R2 = {best_r2:.4f}")

        best_model.fit(X_known, y_known)

        if len(X_missing) > 0:
            y_pred_missing = best_model.predict(X_missing)
            data.loc[missing_mask, target] = y_pred_missing

        best_model_results.append({
            "target": target,
            "best_model": best_model_name,
            "best_r2": best_r2,
            "best_mse": best_mse,
            "best_mae": best_mae,
            "n_known": len(y_known),
            "n_missing": int(np.sum(missing_mask))
        })

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data.to_csv(output_file, index=False)

    base_name = os.path.splitext(output_file)[0]
    all_results_file = f"{base_name}_all_model_results.csv"
    best_results_file = f"{base_name}_best_model_summary.csv"

    pd.DataFrame(all_model_results).to_csv(all_results_file, index=False)
    pd.DataFrame(best_model_results).to_csv(best_results_file, index=False)

    print(f"\\nImputed dataset saved to: {output_file}")
    print(f"All model evaluation results saved to: {all_results_file}")
    print(f"Best model summary saved to: {best_results_file}")


if __name__ == "__main__":
    main()
