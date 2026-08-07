# -*- coding: utf-8 -*-
"""Machine-learning workflow for lanthanide SAC adsorption-energy prediction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

RANDOM_STATE = 42

# Adsorption intermediates listed in the manuscript.
TARGETS = [
    "o", "oh", "h2o", "c", "co2", "cho", "co", "cooh",
    "n", "n2", "n2h", "no", "noh", "h", "f", "s",
]

# Lanthanide descriptors.
ATOMIC_NUMBER = {
    "la": 57, "ce": 58, "pr": 59, "nd": 60, "sm": 62, "eu": 63,
    "gd": 64, "tb": 65, "dy": 66, "ho": 67, "er": 68, "tm": 69,
    "yb": 70, "lu": 71,
}

F_OCCUPANCY = {
    "la": 0, "ce": 1, "pr": 2, "nd": 3, "sm": 5, "eu": 6,
    "gd": 7, "tb": 8, "dy": 9, "ho": 10, "er": 11, "tm": 12,
    "yb": 13, "lu": 14,
}

MODEL_NAMES = [
    "ridge",
    "random_forest",
    "xgboost",
    "svr",
    "hist_gradient_boosting",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOO model evaluation and prediction for lanthanide SAC adsorption energies."
    )
    parser.add_argument("--input-file", required=True, help="Input CSV file.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--feature-set",
        choices=["results", "methods"],
        default="results",
        help=(
            "results: atomic number + substrate configuration + 4f occupancy + atomic radius; "
            "methods: atomic number + binding energy + bonding length + atomic radius."
        ),
    )
    parser.add_argument(
        "--descriptor-file",
        default=None,
        help="CSV containing element and atomic_radius; atomic_number/f_occupancy may also be supplied.",
    )
    parser.add_argument(
        "--bond-length-column",
        default="bond_length",
        help="Bonding-length column used with --feature-set methods.",
    )
    parser.add_argument(
        "--dft-mask-file",
        default=None,
        help="Optional CSV with columns name,target,is_dft for completed datasets.",
    )
    parser.add_argument(
        "--adsorbate-reference-file",
        default=None,
        help="Optional CSV with columns target,adsorbate_energy_eV.",
    )
    parser.add_argument("--min-known", type=int, default=6)
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def add_descriptors(data: pd.DataFrame, descriptor_file: str | None) -> pd.DataFrame:
    data = data.copy()
    data["element"] = data["element"].astype(str).str.lower().str.strip()

    if "atomic_number" not in data.columns:
        data["atomic_number"] = data["element"].map(ATOMIC_NUMBER)
    if "f_occupancy" not in data.columns:
        data["f_occupancy"] = data["element"].map(F_OCCUPANCY)

    if descriptor_file:
        desc = normalize_columns(pd.read_csv(descriptor_file))
        if "element" not in desc.columns or "atomic_radius" not in desc.columns:
            raise ValueError("descriptor-file must contain element and atomic_radius columns.")
        desc["element"] = desc["element"].astype(str).str.lower().str.strip()
        keep = [
            c for c in ["element", "atomic_number", "f_occupancy", "atomic_radius"]
            if c in desc.columns
        ]
        base = data.drop(columns=[c for c in keep if c != "element" and c in data.columns])
        data = base.merge(desc[keep].drop_duplicates("element"), on="element", how="left")

    required = ["atomic_number", "f_occupancy", "atomic_radius"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing descriptor columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Some lanthanide descriptor values are missing.")
    return data


def load_dft_mask(mask_file: str, data: pd.DataFrame, target: str) -> np.ndarray:
    mask = normalize_columns(pd.read_csv(mask_file))
    required = {"name", "target", "is_dft"}
    if not required.issubset(mask.columns):
        raise ValueError("dft-mask-file must contain name,target,is_dft columns.")

    selected = mask[
        mask["target"].astype(str).str.lower().str.strip().eq(target)
        & mask["is_dft"].astype(bool)
    ]
    known_names = set(selected["name"].astype(str))
    return data["name"].astype(str).isin(known_names).to_numpy()


def known_mask(data: pd.DataFrame, target: str, mask_file: str | None) -> np.ndarray:
    if mask_file:
        return load_dft_mask(mask_file, data, target) & data[target].notna().to_numpy()

    mask = data[target].notna().to_numpy()
    if mask.all():
        raise ValueError(
            f"Target '{target}' has no missing entries. Provide --dft-mask-file to identify DFT-calculated samples."
        )
    return mask


def load_adsorbate_references(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    refs = normalize_columns(pd.read_csv(path))
    required = {"target", "adsorbate_energy_ev"}
    if not required.issubset(refs.columns):
        raise ValueError("adsorbate-reference-file must contain target,adsorbate_energy_eV columns.")
    return {
        str(row["target"]).lower().strip(): float(row["adsorbate_energy_ev"])
        for _, row in refs.iterrows()
    }


def feature_columns(data: pd.DataFrame, feature_set: str, bond_length_column: str):
    if feature_set == "results":
        numeric = ["atomic_number", "f_occupancy", "atomic_radius"]
        categorical = ["base_type"]
    else:
        numeric = ["atomic_number", "binding_energy", bond_length_column, "atomic_radius"]
        categorical = []

    missing = [c for c in numeric + categorical if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return numeric, categorical


def make_preprocessor(numeric, categorical, scale_numeric: bool):
    transformers = []
    if numeric:
        transformers.append(
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric)
        )
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    return ColumnTransformer(transformers, remainder="drop")


# Model settings used for comparison.
def create_model(name: str, numeric, categorical):
    if name == "ridge":
        return Pipeline([
            ("pre", make_preprocessor(numeric, categorical, True)),
            ("model", Ridge(alpha=1.0)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("pre", make_preprocessor(numeric, categorical, False)),
            ("model", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ])
    if name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed.")
        return Pipeline([
            ("pre", make_preprocessor(numeric, categorical, False)),
            ("model", XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            )),
        ])
    if name == "svr":
        return Pipeline([
            ("pre", make_preprocessor(numeric, categorical, True)),
            ("model", SVR(C=10, kernel="rbf")),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("pre", make_preprocessor(numeric, categorical, False)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ])
    raise ValueError(name)


def build_target(data: pd.DataFrame, target: str, adsorbate_refs: dict[str, float]) -> np.ndarray:
    # Adsorption-energy target.
    y = data[target].astype(float) - data["re_atom_energy"].astype(float) - data["base_c"].astype(float)
    if target in adsorbate_refs:
        y = y - adsorbate_refs[target]
    return y.to_numpy(float)


def safe_corr(func, y_true, y_pred):
    try:
        return float(func(y_true, y_pred).statistic)
    except Exception:
        return np.nan


def evaluate_model(model_name, X, y, numeric, categorical):
    # Leave-one-out cross-validation.
    loo = LeaveOneOut()
    pred = np.full(len(y), np.nan, dtype=float)

    for train_idx, test_idx in loo.split(X):
        model = create_model(model_name, numeric, categorical)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])

    mse = mean_squared_error(y, pred)
    metrics = {
        "r2": float(r2_score(y, pred)),
        "mse_eV2": float(mse),
        "rmse_eV": float(math.sqrt(mse)),
        "mae_eV": float(mean_absolute_error(y, pred)),
        "pearson_r": safe_corr(pearsonr, y, pred),
        "spearman_rho": safe_corr(spearmanr, y, pred),
    }
    return pred, metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = normalize_columns(pd.read_csv(args.input_file))
    required_source = ["name", "element", "base_c", "re_atom_energy", "binding_energy"]
    missing_source = [c for c in required_source if c not in data.columns]
    if missing_source:
        raise ValueError(f"Missing source columns: {missing_source}")

    data = add_descriptors(data, args.descriptor_file)
    adsorbate_refs = load_adsorbate_references(args.adsorbate_reference_file)
    numeric, categorical = feature_columns(
        data, args.feature_set, args.bond_length_column.lower()
    )
    features = numeric + categorical

    metric_rows = []
    prediction_rows = []
    best_rows = []
    completed = data.copy()

    for target in TARGETS:
        if target not in data.columns:
            continue

        is_known = known_mask(data, target, args.dft_mask_file)
        n_known = int(is_known.sum())
        if n_known < args.min_known:
            continue

        known = data.loc[is_known].copy()
        X_known = known[features].copy()
        y_known = build_target(known, target, adsorbate_refs)

        best_name = None
        best_r2 = -np.inf
        best_metrics = None

        for model_name in MODEL_NAMES:
            try:
                pred, metrics = evaluate_model(
                    model_name, X_known, y_known, numeric, categorical
                )
            except Exception as exc:
                metric_rows.append({
                    "target": target,
                    "model": model_name,
                    "n_dft": n_known,
                    "error": str(exc),
                })
                continue

            metric_rows.append({
                "target": target,
                "model": model_name,
                "n_dft": n_known,
                **metrics,
            })

            for (_, row), y_true, y_pred in zip(known.iterrows(), y_known, pred):
                prediction_rows.append({
                    "target": target,
                    "model": model_name,
                    "name": row["name"],
                    "element": row["element"],
                    "y_true_eV": float(y_true),
                    "y_pred_eV": float(y_pred),
                    "residual_eV": float(y_pred - y_true),
                })

            if np.isfinite(metrics["r2"]) and metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = model_name
                best_metrics = metrics

        if best_name is None:
            continue

        best_rows.append({
            "target": target,
            "best_model": best_name,
            "n_dft": n_known,
            **best_metrics,
        })

        # Fit the selected model on all DFT-known samples and predict the remaining entries.
        model = create_model(best_name, numeric, categorical)
        model.fit(X_known, y_known)
        to_predict = ~is_known

        if to_predict.any():
            X_pred = data.loc[to_predict, features].copy()
            y_pred = model.predict(X_pred)
            ref = adsorbate_refs.get(target, 0.0)
            total_pred = (
                y_pred
                + data.loc[to_predict, "re_atom_energy"].to_numpy(float)
                + data.loc[to_predict, "base_c"].to_numpy(float)
                + ref
            )
            completed.loc[to_predict, target] = total_pred

    # Save the completed dataset and model-evaluation tables.
    pd.DataFrame(metric_rows).to_csv(output_dir / "all_model_metrics.csv", index=False)
    pd.DataFrame(best_rows).to_csv(output_dir / "best_model_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "loo_predictions.csv", index=False)
    completed.to_csv(output_dir / "predicted_dataset.csv", index=False)

    if best_rows:
        summary = pd.DataFrame(best_rows)
        columns = ["target", "best_model", "n_dft", "r2", "rmse_eV", "mae_eV", "pearson_r"]
        print(summary[columns].to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
