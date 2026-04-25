
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probabilistic and Bayesian Models for Agricultural Decision Support
-----------------------------------------------------------------

This script implements a fully reproducible computational benchmark aligned with
a research article on probabilistic and Bayesian decision support for agriculture
under uncertainty, data scarcity, and climate risk.

The program creates three result subfolders named exactly:

    5.1
    5.2
    5.3

The folders correspond to the three analytical subsections of the article's
results section:

    5.1 Yield forecasting under data scarcity
    5.2 Sequential irrigation support under evolving evidence
    5.3 Pest risk, intervention utility, and value of information

Why the script uses synthetic data
----------------------------------
The objective of the article is methodological and architectural rather than
site-specific. For that reason, the code generates an agronomically constrained
synthetic benchmark. This allows the whole workflow to be executed offline, with
no hidden dependencies on private datasets, while preserving the structural
properties that matter for uncertainty-aware agricultural decision support:
heterogeneous regions, scarce historical observations, noisy sensors, climate
variability, sequential evidence, and action-dependent outcomes.

High-level structure of the script
----------------------------------
1. Generate a multiregional yield dataset with scarce-data regions.
2. Fit and evaluate:
      - a pooled deterministic linear model,
      - a local no-pooling deterministic model,
      - a hierarchical Bayesian regression estimated by Gibbs sampling.
3. Simulate a sequential irrigation problem with three policies:
      - static calendar irrigation,
      - threshold irrigation from noisy observations,
      - Bayesian risk-aware irrigation using recursive state updating.
4. Simulate a pest and outbreak management problem with:
      - a deterministic prior-threshold rule,
      - a Bayesian posterior rule after scouting,
      - a Bayesian value-of-information policy that decides when an
        additional scouting action is economically justified.
5. Save tables, figures, and machine-readable summaries to the required
   subsection folders.
6. Package all generated outputs into a ZIP archive.

The code is intentionally verbose and heavily commented so that it can function
both as a runnable artifact and as a transparent methodological appendix.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# 1) GLOBAL CONFIGURATION
# =============================================================================

# A fixed seed is used throughout the script so that every run is deterministic
# and reproducible. Reproducibility is especially important in a methodological
# article because figures, tables, and narrative interpretation should remain
# stable across executions.
GLOBAL_SEED = 42

# Result directory names are fixed by the article structure requested by the user.
SUBFOLDERS = ["5.1", "5.2", "5.3"]

# Matplotlib defaults are intentionally simple and publication-oriented.
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# =============================================================================
# 2) SMALL UTILITY HELPERS
# =============================================================================

def ensure_clean_dir(path: Path) -> None:
    """
    Remove an existing directory and create a fresh replacement.

    This is useful when the script is rerun many times during experimentation:
    stale figures from previous runs can otherwise contaminate the current
    outputs and create inconsistencies between tables and charts.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path, round_digits: int = 4) -> None:
    """
    Save a dataframe to CSV with a stable floating-point representation.

    A rounded export improves readability when the tables are later inserted
    into a manuscript.
    """
    export_df = df.copy()
    float_cols = export_df.select_dtypes(include=["float", "float64", "float32"]).columns
    export_df[float_cols] = export_df[float_cols].round(round_digits)
    export_df.to_csv(path, index=False)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Compute the binary Shannon entropy of Bernoulli probabilities.

    Entropy is used in subsection 5.3 to quantify posterior uncertainty and to
    show where an additional scouting action has the highest value.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """
    Compute the Brier score for probabilistic binary predictions.

    Lower values indicate better calibrated and more accurate probabilities.
    """
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


# =============================================================================
# 3) SUBSECTION 5.1 — YIELD FORECASTING UNDER DATA SCARCITY
# =============================================================================

def generate_yield_dataset(
    seed: int = GLOBAL_SEED,
    n_regions: int = 8,
    farms_per_region: int = 12,
    seasons: int = 5,
    plots_per_farm_season: int = 3,
) -> pd.DataFrame:
    """
    Create a synthetic yield forecasting dataset with multilevel heterogeneity.

    Conceptual design
    -----------------
    The synthetic dataset is designed to reproduce a common agricultural
    forecasting problem:
        - multiple regions with different agro-climatic baselines,
        - repeated observations across seasons,
        - farms with heterogeneous soils and management conditions,
        - noisy and incomplete local information,
        - a subset of regions with scarce historical data.

    The outcome variable is yield in tonnes per hectare. Predictors include
    climatic stress, soil quality, vegetation condition, nutrient intensity,
    management quality, pest pressure, and irrigation level.

    The data generating process is intentionally nonlinear in places because
    real agricultural systems rarely behave as perfect linear systems. The
    fitted hierarchical Bayesian model remains linear for interpretability,
    which makes the forecasting task nontrivial and realistic.
    """
    rng = np.random.default_rng(seed)

    regions = [f"Region_{i + 1}" for i in range(n_regions)]
    scarce_regions = set(regions[-3:])

    region_base_rain = {r: rng.normal(430, 50) for r in regions}
    region_base_temp = {r: rng.normal(0.0, 0.6) for r in regions}
    region_soil_bonus = {r: rng.normal(0.0, 0.6) for r in regions}
    region_market_noise = {r: rng.normal(0.0, 0.15) for r in regions}

    rows: List[Dict[str, float]] = []
    farm_counter = 0

    for region in regions:
        for _ in range(farms_per_region):
            farm_counter += 1
            farm_id = f"F{farm_counter:03d}"

            soil_organic = np.clip(
                rng.normal(2.2 + 0.25 * region_soil_bonus[region], 0.45), 1.0, 4.2
            )
            water_holding_capacity = np.clip(
                rng.normal(160 + 12 * region_soil_bonus[region], 18), 110, 220
            )
            management_index = np.clip(rng.beta(4, 2), 0.25, 0.98)
            sensor_quality = np.clip(rng.beta(5, 2), 0.3, 0.99)

            for season in range(1, seasons + 1):
                season_rain_anomaly = rng.normal(0, 70)
                season_heat = np.clip(
                    rng.normal(18 + 2.0 * season + 2.8 * region_base_temp[region], 4.5),
                    5,
                    40,
                )
                season_policy_shock = rng.normal(0, 0.15)

                for _ in range(plots_per_farm_season):
                    rainfall = np.clip(
                        region_base_rain[region] + season_rain_anomaly + rng.normal(0, 35),
                        180,
                        760,
                    )
                    heat_days = np.clip(season_heat + rng.normal(0, 2.0), 4, 45)
                    ndvi_peak = np.clip(
                        0.42
                        + 0.00055 * rainfall
                        - 0.006 * heat_days
                        + 0.06 * management_index
                        + rng.normal(0, 0.045),
                        0.28,
                        0.92,
                    )
                    nitrogen = np.clip(rng.normal(135 + 15 * management_index, 22), 60, 210)
                    cultivar_resilience = np.clip(
                        rng.normal(1.0 + 0.04 * management_index, 0.07), 0.78, 1.22
                    )
                    early_pest_pressure = np.clip(
                        0.25
                        + 0.014 * (heat_days - 20)
                        - 0.00045 * (rainfall - 420)
                        - 0.18 * management_index
                        + rng.normal(0, 0.08),
                        0.01,
                        0.95,
                    )
                    irrigation = np.clip(
                        max(
                            0,
                            100
                            - 0.18 * (rainfall - 380)
                            + 0.55 * (heat_days - 18)
                            - 12 * soil_organic
                            - 0.12 * water_holding_capacity
                            + rng.normal(0, 8),
                        ),
                        0,
                        220,
                    )
                    forecast_rain_prob = np.clip(
                        0.15 + 0.0012 * rainfall - 0.003 * heat_days + rng.normal(0, 0.08),
                        0.02,
                        0.98,
                    )
                    forecast_rain_prob = min(max(forecast_rain_prob / 1.2, 0.02), 0.98)

                    # Yield data generating process.
                    # A mixture of linear and mildly nonlinear effects is used so that
                    # uncertainty-aware models have a meaningful task.
                    yield_t_ha = (
                        5.0
                        + 0.0055 * (rainfall - 400)
                        - 0.09 * (heat_days - 20)
                        + 0.62 * (soil_organic - 2.0)
                        + 0.0075 * (water_holding_capacity - 160)
                        + 4.0 * (ndvi_peak - 0.5)
                        + 0.007 * (nitrogen - 120)
                        - 0.000035 * (nitrogen - 145) ** 2
                        + 1.35 * (management_index - 0.6)
                        + 0.8 * (cultivar_resilience - 1.0)
                        - 1.6 * early_pest_pressure
                        + 0.004 * irrigation
                        + region_soil_bonus[region]
                        + 0.18 * season
                        + region_market_noise[region]
                        + season_policy_shock
                        + rng.normal(0, 0.42 + 0.18 * (1 - sensor_quality))
                    )
                    yield_t_ha = np.clip(yield_t_ha, 1.0, 11.5)

                    rows.append(
                        {
                            "region": region,
                            "farm_id": farm_id,
                            "season": season,
                            "rainfall_mm": rainfall,
                            "heat_days": heat_days,
                            "soil_organic_matter": soil_organic,
                            "water_holding_capacity": water_holding_capacity,
                            "ndvi_peak": ndvi_peak,
                            "nitrogen_kg_ha": nitrogen,
                            "management_index": management_index,
                            "sensor_quality": sensor_quality,
                            "cultivar_resilience": cultivar_resilience,
                            "early_pest_pressure": early_pest_pressure,
                            "irrigation_mm": irrigation,
                            "forecast_rain_prob": forecast_rain_prob,
                            "yield_t_ha": yield_t_ha,
                            "scarce_region": region in scarce_regions,
                        }
                    )

    return pd.DataFrame(rows)


def split_yield_data(
    df: pd.DataFrame,
    test_season: int = 5,
    scarce_train_fraction: float = 0.18,
    seed: int = GLOBAL_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the yield dataset into training and testing partitions.

    Design choice
    -------------
    The last season is reserved for testing so that the evaluation reflects an
    operational forecasting setting: the model learns from historical seasons and
    predicts a future season.

    To emulate data scarcity, only a small fraction of historical observations is
    retained for the designated scarce-data regions.
    """
    rng = np.random.default_rng(seed)

    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()

    keep_idx: List[int] = []
    for region, group in train.groupby("region"):
        if bool(group["scarce_region"].iloc[0]):
            n_keep = max(12, int(len(group) * scarce_train_fraction))
            keep_idx.extend(rng.choice(group.index.to_numpy(), size=n_keep, replace=False))
        else:
            keep_idx.extend(group.index.to_list())

    train = train.loc[sorted(keep_idx)].copy().reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test


def standardize_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Standardize features with training-set statistics only.

    This prevents information leakage from the test season into the fitted
    models, which is a crucial requirement for trustworthy forecasting.
    """
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1.0)
    x_train = ((train_df[feature_cols] - means) / stds).to_numpy()
    x_test = ((test_df[feature_cols] - means) / stds).to_numpy()
    return x_train, x_test, means, stds


def gibbs_hierarchical_linear(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_iter: int = 2000,
    burn: int = 700,
    thin: int = 4,
    seed: int = GLOBAL_SEED,
) -> Dict[str, np.ndarray]:
    """
    Estimate a hierarchical Bayesian linear model by Gibbs sampling.

    Model
    -----
    y_i = alpha_{g[i]} + x_i' beta + epsilon_i
    epsilon_i ~ Normal(0, sigma^2)

    alpha_r ~ Normal(mu_alpha, tau^2)

    Why this model matters
    ----------------------
    The random regional intercepts allow the model to "borrow strength" across
    regions. This is exactly the mechanism that should be helpful when some
    regions have very few historical observations.

    Why Gibbs sampling
    ------------------
    The chosen priors are conjugate, which makes Gibbs sampling transparent and
    computationally efficient for a fully reproducible offline benchmark.
    """
    rng = np.random.default_rng(seed)

    n_obs, n_features = x.shape

    # Weakly informative priors.
    v_beta = 10.0
    m0 = 0.0
    v0 = 10.0
    a_sigma = 2.0
    b_sigma = 1.0
    a_tau = 2.0
    b_tau = 1.0

    beta = np.zeros(n_features)
    alpha = np.repeat(np.mean(y), n_groups)
    mu_alpha = float(np.mean(y))
    sigma2 = float(np.var(y))
    tau2 = float(np.var(alpha - mu_alpha) + 0.1)

    xtx = x.T @ x
    identity = np.eye(n_features)
    group_indices = [np.where(groups == r)[0] for r in range(n_groups)]

    samples = {"beta": [], "alpha": [], "mu_alpha": [], "sigma2": [], "tau2": []}

    for iteration in range(n_iter):
        # ---- Update the regression coefficients beta.
        y_tilde = y - alpha[groups]
        v_beta_post = np.linalg.inv(xtx / sigma2 + identity / v_beta)
        m_beta_post = v_beta_post @ (x.T @ y_tilde / sigma2)
        beta = rng.multivariate_normal(m_beta_post, v_beta_post)

        # ---- Update the random intercept alpha for each region.
        for r in range(n_groups):
            idx = group_indices[r]
            n_r = len(idx)
            residual_r = y[idx] - x[idx] @ beta
            var_r = 1.0 / (n_r / sigma2 + 1.0 / tau2)
            mean_r = var_r * (residual_r.sum() / sigma2 + mu_alpha / tau2)
            alpha[r] = rng.normal(mean_r, math.sqrt(var_r))

        # ---- Update the population mean of the random intercepts.
        v_mu_post = 1.0 / (n_groups / tau2 + 1.0 / v0)
        m_mu_post = v_mu_post * (alpha.sum() / tau2 + m0 / v0)
        mu_alpha = rng.normal(m_mu_post, math.sqrt(v_mu_post))

        # ---- Update observational variance sigma^2.
        residual = y - x @ beta - alpha[groups]
        shape_sigma = a_sigma + n_obs / 2.0
        scale_sigma = b_sigma + 0.5 * float(np.dot(residual, residual))
        sigma2 = 1.0 / rng.gamma(shape_sigma, 1.0 / scale_sigma)

        # ---- Update between-region variance tau^2.
        alpha_centered = alpha - mu_alpha
        shape_tau = a_tau + n_groups / 2.0
        scale_tau = b_tau + 0.5 * float(np.dot(alpha_centered, alpha_centered))
        tau2 = 1.0 / rng.gamma(shape_tau, 1.0 / scale_tau)

        if iteration >= burn and ((iteration - burn) % thin == 0):
            samples["beta"].append(beta.copy())
            samples["alpha"].append(alpha.copy())
            samples["mu_alpha"].append(mu_alpha)
            samples["sigma2"].append(sigma2)
            samples["tau2"].append(tau2)

    for key in samples:
        samples[key] = np.asarray(samples[key])

    return samples


def predict_hierarchical(
    samples: Dict[str, np.ndarray],
    x_new: np.ndarray,
    groups_new: np.ndarray,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate posterior predictive draws from the hierarchical Bayesian model.

    The posterior predictive distribution is more informative than a point
    prediction because it preserves uncertainty from both parameter estimation
    and irreducible observational variability.
    """
    rng = np.random.default_rng(seed)

    n_draws = len(samples["sigma2"])
    n_cases = x_new.shape[0]
    draws = np.zeros((n_draws, n_cases), dtype=float)

    for s in range(n_draws):
        mean_component = x_new @ samples["beta"][s] + samples["alpha"][s][groups_new]
        draws[s, :] = rng.normal(mean_component, np.sqrt(samples["sigma2"][s]))

    pred_mean = draws.mean(axis=0)
    return pred_mean, draws


def fit_pooled_linear(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int = 123,
) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    """
    Fit a single deterministic regression to all regions pooled together.

    This model ignores regional heterogeneity and therefore acts as a sensible
    but deliberately limited baseline.
    """
    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df["yield_t_ha"].to_numpy()
    x_test = test_df[feature_cols].to_numpy()

    model = LinearRegression().fit(x_train, y_train)
    pred_train = model.predict(x_train)

    # Residual standard deviation is used to derive approximate predictive
    # distributions. This is not a full probabilistic model, but it allows a
    # consistent interval comparison against the Bayesian model.
    resid_std = np.std(y_train - pred_train, ddof=x_train.shape[1] + 1)
    pred_mean = model.predict(x_test)

    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=pred_mean, scale=resid_std, size=(500, len(pred_mean)))
    return model, pred_mean, draws


def fit_local_linear(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a separate deterministic regression inside each region.

    This baseline preserves heterogeneity but does not borrow information across
    regions. It is therefore especially vulnerable when training data are scarce.
    """
    pred_mean = np.zeros(len(test_df), dtype=float)
    draws = np.zeros((500, len(test_df)), dtype=float)

    for region, idx in test_df.groupby("region").groups.items():
        train_region = train_df[train_df["region"] == region]
        test_region = test_df.loc[idx]

        x_train = train_region[feature_cols].to_numpy()
        y_train = train_region["yield_t_ha"].to_numpy()
        x_test = test_region[feature_cols].to_numpy()

        if len(train_region) >= len(feature_cols) + 2:
            model = LinearRegression().fit(x_train, y_train)
            region_pred = model.predict(x_test)
            resid_std = np.std(y_train - model.predict(x_train), ddof=min(len(feature_cols) + 1, len(y_train) - 1))
            resid_std = max(float(resid_std), 0.35)
        elif len(train_region) >= 5:
            # If there are very few samples, a local mean is more stable than
            # an overparameterized local regression.
            region_pred = np.full(len(test_region), y_train.mean())
            resid_std = float(np.std(y_train, ddof=1)) if len(y_train) > 1 else 0.8
        else:
            region_pred = np.full(len(test_region), train_df["yield_t_ha"].mean())
            resid_std = float(train_df["yield_t_ha"].std())

        pred_mean[idx] = region_pred
        rng = np.random.default_rng(abs(hash(region)) % (2**32))
        draws[:, idx] = rng.normal(loc=region_pred, scale=resid_std, size=(500, len(test_region)))

    return pred_mean, draws


def interval_metrics(
    y_true: np.ndarray, draws: np.ndarray, alpha: float = 0.10
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute interval-based probabilistic forecast diagnostics.

    Returned metrics
    ----------------
    coverage:
        empirical coverage of the prediction interval.
    width:
        mean interval width.
    interval_score:
        proper interval score that rewards narrow intervals and penalizes
        intervals that miss the observation.
    """
    lower = np.quantile(draws, alpha / 2.0, axis=0)
    upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    penalty = np.where(
        y_true < lower,
        (2.0 / alpha) * (lower - y_true),
        np.where(y_true > upper, (2.0 / alpha) * (y_true - upper), 0.0),
    )
    interval_score = float(np.mean((upper - lower) + penalty))
    return coverage, width, interval_score, lower, upper


def summarize_regression_model(y_true: np.ndarray, pred_mean: np.ndarray, draws: np.ndarray) -> Dict[str, object]:
    """
    Gather deterministic and probabilistic metrics in one dictionary.
    """
    coverage, width, interval_score, lower, upper = interval_metrics(y_true, draws, alpha=0.10)
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, pred_mean)),
        "MAE": mean_absolute_error(y_true, pred_mean),
        "Coverage_90": coverage,
        "Mean_Width_90": width,
        "Interval_Score_90": interval_score,
        "Pred_Mean": pred_mean,
        "Lower_90": lower,
        "Upper_90": upper,
    }


def empirical_coverage_curve(
    y_true: np.ndarray,
    draws: np.ndarray,
    nominal_levels: Iterable[float] = np.arange(0.50, 0.96, 0.05),
) -> pd.DataFrame:
    """
    Estimate how well nominal interval levels match empirical coverage.

    If a model is perfectly calibrated, the empirical coverage curve should
    follow the 45-degree line.
    """
    empirical = []
    widths = []
    for level in nominal_levels:
        alpha = 1.0 - level
        lower = np.quantile(draws, alpha / 2.0, axis=0)
        upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
        empirical.append(np.mean((y_true >= lower) & (y_true <= upper)))
        widths.append(np.mean(upper - lower))
    return pd.DataFrame({"Nominal": list(nominal_levels), "Empirical": empirical, "Mean_Width": widths})


def run_yield_module(output_dir: Path) -> Dict[str, object]:
    """
    Execute subsection 5.1 end-to-end and save all artifacts.
    """
    subdir = output_dir / "5.1"
    subdir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "rainfall_mm",
        "heat_days",
        "soil_organic_matter",
        "water_holding_capacity",
        "ndvi_peak",
        "nitrogen_kg_ha",
        "management_index",
        "cultivar_resilience",
        "early_pest_pressure",
        "irrigation_mm",
    ]

    df = generate_yield_dataset(seed=GLOBAL_SEED)
    train_df, test_df = split_yield_data(df, test_season=5, scarce_train_fraction=0.18, seed=GLOBAL_SEED)

    x_train, x_test, _, _ = standardize_features(train_df, test_df, feature_cols)
    regions = sorted(train_df["region"].unique())
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    g_train = train_df["region"].map(region_to_idx).to_numpy()
    g_test = test_df["region"].map(region_to_idx).to_numpy()
    y_test = test_df["yield_t_ha"].to_numpy()

    # Baselines.
    _, pooled_mean, pooled_draws = fit_pooled_linear(train_df, test_df, feature_cols)
    local_mean, local_draws = fit_local_linear(train_df, test_df, feature_cols)

    # Hierarchical Bayesian model.
    bayes_samples = gibbs_hierarchical_linear(
        x_train,
        train_df["yield_t_ha"].to_numpy(),
        g_train,
        n_groups=len(regions),
        n_iter=2000,
        burn=700,
        thin=4,
        seed=GLOBAL_SEED,
    )
    bayes_mean, bayes_draws = predict_hierarchical(bayes_samples, x_test, g_test, seed=123)

    pooled_summary = summarize_regression_model(y_test, pooled_mean, pooled_draws)
    local_summary = summarize_regression_model(y_test, local_mean, local_draws)
    bayes_summary = summarize_regression_model(y_test, bayes_mean, bayes_draws)

    table_1 = pd.DataFrame(
        [
            {
                "Model": "Pooled Linear",
                "RMSE": pooled_summary["RMSE"],
                "MAE": pooled_summary["MAE"],
                "Coverage_90": pooled_summary["Coverage_90"],
                "Mean_Width_90": pooled_summary["Mean_Width_90"],
                "Interval_Score_90": pooled_summary["Interval_Score_90"],
            },
            {
                "Model": "Local Linear",
                "RMSE": local_summary["RMSE"],
                "MAE": local_summary["MAE"],
                "Coverage_90": local_summary["Coverage_90"],
                "Mean_Width_90": local_summary["Mean_Width_90"],
                "Interval_Score_90": local_summary["Interval_Score_90"],
            },
            {
                "Model": "Hierarchical Bayesian",
                "RMSE": bayes_summary["RMSE"],
                "MAE": bayes_summary["MAE"],
                "Coverage_90": bayes_summary["Coverage_90"],
                "Mean_Width_90": bayes_summary["Mean_Width_90"],
                "Interval_Score_90": bayes_summary["Interval_Score_90"],
            },
        ]
    )
    save_dataframe(table_1, subdir / "Table_1_yield_model_comparison.csv")

    rows = []
    for subset_name, mask in [
        ("Data-rich regions", ~test_df["scarce_region"].to_numpy()),
        ("Scarce-data regions", test_df["scarce_region"].to_numpy()),
    ]:
        y_subset = y_test[mask]
        for model_name, pred_mean, draws in [
            ("Pooled Linear", pooled_mean, pooled_draws),
            ("Local Linear", local_mean, local_draws),
            ("Hierarchical Bayesian", bayes_mean, bayes_draws),
        ]:
            coverage, width, interval_score, _, _ = interval_metrics(y_subset, draws[:, mask], alpha=0.10)
            rows.append(
                {
                    "Subset": subset_name,
                    "Model": model_name,
                    "RMSE": math.sqrt(mean_squared_error(y_subset, pred_mean[mask])),
                    "MAE": mean_absolute_error(y_subset, pred_mean[mask]),
                    "Coverage_90": coverage,
                    "Mean_Width_90": width,
                    "Interval_Score_90": interval_score,
                }
            )
    table_2 = pd.DataFrame(rows)
    save_dataframe(table_2, subdir / "Table_2_yield_subset_comparison.csv")

    # ---- Figure 1: observed vs predicted under the hierarchical Bayesian model.
    plt.figure(figsize=(7.2, 5.6))
    rich_mask = ~test_df["scarce_region"].to_numpy()
    scarce_mask = test_df["scarce_region"].to_numpy()
    plt.scatter(y_test[rich_mask], bayes_mean[rich_mask], alpha=0.65, label="Data-rich regions")
    plt.scatter(y_test[scarce_mask], bayes_mean[scarce_mask], alpha=0.75, marker="s", label="Scarce-data regions")
    lo = min(y_test.min(), bayes_mean.min()) - 0.2
    hi = max(y_test.max(), bayes_mean.max()) + 0.2
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Ideal agreement")
    plt.xlabel("Observed yield (t/ha)")
    plt.ylabel("Predicted mean yield (t/ha)")
    plt.title("Figure 1. Hierarchical Bayesian predictions versus observed yield")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_1_yield_predictions_hierarchical_bayesian.png", bbox_inches="tight")
    plt.close()

    # ---- Figure 2: interval calibration.
    pooled_cal = empirical_coverage_curve(y_test, pooled_draws)
    local_cal = empirical_coverage_curve(y_test, local_draws)
    bayes_cal = empirical_coverage_curve(y_test, bayes_draws)

    plt.figure(figsize=(7.0, 5.4))
    plt.plot(pooled_cal["Nominal"], pooled_cal["Empirical"], marker="o", label="Pooled Linear")
    plt.plot(local_cal["Nominal"], local_cal["Empirical"], marker="o", label="Local Linear")
    plt.plot(bayes_cal["Nominal"], bayes_cal["Empirical"], marker="o", label="Hierarchical Bayesian")
    plt.plot([0.5, 0.95], [0.5, 0.95], linestyle="--", linewidth=1.2, label="Ideal calibration")
    plt.xlabel("Nominal interval level")
    plt.ylabel("Empirical coverage")
    plt.title("Figure 2. Prediction-interval calibration across models")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_2_yield_interval_calibration.png", bbox_inches="tight")
    plt.close()

    detailed_predictions = test_df.copy()
    detailed_predictions["pooled_mean"] = pooled_mean
    detailed_predictions["local_mean"] = local_mean
    detailed_predictions["bayes_mean"] = bayes_mean
    detailed_predictions["bayes_lower_90"] = bayes_summary["Lower_90"]
    detailed_predictions["bayes_upper_90"] = bayes_summary["Upper_90"]
    save_dataframe(detailed_predictions, subdir / "yield_test_predictions_detailed.csv")

    module_summary = {
        "overall_model_table": table_1.round(6).to_dict(orient="records"),
        "subset_model_table": table_2.round(6).to_dict(orient="records"),
    }
    (subdir / "yield_module_summary.json").write_text(json.dumps(module_summary, indent=2), encoding="utf-8")

    return {
        "data": df,
        "train": train_df,
        "test": test_df,
        "table_1": table_1,
        "table_2": table_2,
        "pooled_summary": pooled_summary,
        "local_summary": local_summary,
        "bayes_summary": bayes_summary,
        "pooled_cal": pooled_cal,
        "local_cal": local_cal,
        "bayes_cal": bayes_cal,
    }


# =============================================================================
# 4) SUBSECTION 5.2 — SEQUENTIAL IRRIGATION SUPPORT
# =============================================================================

IRRIGATION_PARAMS = {
    "bayes_p": 0.39,
    "bayes_hi": 0.62,
    "bayes_water": 18.0,
    "thresh_level": 90.0,
    "static_every": 7,
}


def simulate_irrigation_field(
    policy: str,
    seed: int,
    days: int = 110,
    bayes_p: float = 0.39,
    bayes_hi: float = 0.62,
    bayes_water: float = 18.0,
    thresh_level: float = 90.0,
    static_every: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate one field-season irrigation problem.

    State variable
    --------------
    The hidden state is soil moisture deficit in millimetres. Larger deficit means
    more water stress. The Bayesian policy recursively updates this hidden state
    from noisy measurements and short-horizon weather expectations.

    Why this setup is useful
    ------------------------
    Irrigation decisions are made sequentially, under uncertainty, and each action
    directly changes the future state of the system. This makes the problem a good
    example of why posterior updating matters for agricultural decisions.
    """
    rng = np.random.default_rng(seed)

    capacity = float(np.clip(rng.normal(170, 18), 130, 220))
    stress_threshold = 95.0
    start_deficit = float(np.clip(rng.normal(40, 12), 10, 80))
    sensor_sd = float(np.clip(rng.normal(9.0, 1.5), 5.5, 13))
    process_sd = 5.0

    # Initial Bayesian belief about the hidden state.
    posterior_mean = start_deficit + rng.normal(0, 7)
    posterior_var = 36.0

    true_deficit = start_deficit

    records: List[Dict[str, float]] = []
    total_water = 0.0
    stress_days = 0
    severe_stress_days = 0
    irrigation_events = 0

    for day in range(1, days + 1):
        stage = day / days

        et = np.clip(3.0 + 2.8 * np.sin(np.pi * stage) + rng.normal(0, 0.7), 1.8, 7.4)
        rain_occurs = rng.random() < (0.22 + 0.10 * np.cos(2 * np.pi * stage))
        rain = float(min(rng.gamma(1.7, 8.0), 38.0)) if rain_occurs else 0.0

        # Imperfect short-horizon forecasts.
        rain_fore_mean = max(0.0, rain + rng.normal(0, 8.5)) * 0.5 + max(0, rng.normal(4, 4))
        rain_fore_var = 24.0
        et_fore = float(np.clip(et + rng.normal(0.3, 0.6), 1.5, 7.8))

        observation = true_deficit + rng.normal(0, sensor_sd)

        irrigation = 0.0
        p_stress_next3 = np.nan
        decision_score = np.nan
        posterior_mean_for_record = np.nan
        posterior_sd_for_record = np.nan

        if policy == "static":
            if 22 <= day <= 96 and (day - 22) % static_every == 0:
                irrigation = 18.0

        elif policy == "threshold":
            if observation > thresh_level:
                irrigation = 18.0

        elif policy == "bayesian":
            # ---- Measurement update (Bayesian filtering step).
            kalman_gain = posterior_var / (posterior_var + sensor_sd**2)
            posterior_mean = posterior_mean + kalman_gain * (observation - posterior_mean)
            posterior_var = (1.0 - kalman_gain) * posterior_var

            # ---- Short-horizon predictive risk calculation.
            horizon = 3
            pred_mean = posterior_mean + horizon * (et_fore - rain_fore_mean / 3.0)
            pred_var = posterior_var + horizon * (process_sd**2) + rain_fore_var

            p_stress_next3 = float(
                1.0 - st.norm.cdf(stress_threshold, loc=pred_mean, scale=np.sqrt(pred_var))
            )

            # ---- Simple expected-utility decision layer.
            crop_value = 200.0
            avoided_loss = crop_value * (0.12 + 0.24 * np.sin(np.pi * stage)) * max(p_stress_next3, 0.0)
            water_cost = 1.5 * bayes_water
            environmental_cost = 6.0
            decision_score = float(avoided_loss - (water_cost + environmental_cost))

            if (p_stress_next3 > bayes_p and decision_score > -2.0) or (p_stress_next3 > bayes_hi):
                irrigation = bayes_water

            posterior_mean_for_record = posterior_mean
            posterior_sd_for_record = math.sqrt(posterior_var)

        else:
            raise ValueError(f"Unknown irrigation policy: {policy}")

        if irrigation > 0:
            irrigation_events += 1
            total_water += irrigation

        # ---- True system transition.
        true_deficit = float(
            np.clip(true_deficit + et - rain - irrigation + rng.normal(0, process_sd), 0.0, capacity)
        )
        stress = int(true_deficit > stress_threshold)
        severe_stress = int(true_deficit > stress_threshold + 22.0)
        stress_days += stress
        severe_stress_days += severe_stress

        # ---- Forecast propagation for the next day.
        if policy == "bayesian":
            posterior_mean = posterior_mean + et_fore - rain_fore_mean / 3.0 - irrigation
            posterior_var = posterior_var + process_sd**2

        records.append(
            {
                "day": day,
                "policy": policy,
                "capacity": capacity,
                "stress_threshold": stress_threshold,
                "true_deficit": true_deficit,
                "observation": observation,
                "irrigation_mm": irrigation,
                "stress": stress,
                "severe_stress": severe_stress,
                "posterior_mean": posterior_mean_for_record,
                "posterior_sd": posterior_sd_for_record,
                "p_stress_next3": p_stress_next3,
                "decision_score": decision_score,
                "et": et,
                "rain": rain,
            }
        )

    # Seasonal summary metrics.
    yield_proxy = max(
        2.8,
        7.4 - 0.048 * stress_days - 0.08 * severe_stress_days - 0.0023 * max(total_water - 165, 0),
    )
    water_productivity = yield_proxy / max(total_water, 1.0)
    utility_index = 220 * yield_proxy - 1.5 * total_water - 9.5 * severe_stress_days - 4.5 * stress_days

    summary = {
        "total_water_mm": total_water,
        "stress_days": stress_days,
        "severe_stress_days": severe_stress_days,
        "yield_proxy_t_ha": yield_proxy,
        "water_productivity": water_productivity,
        "utility_index": utility_index,
        "irrigation_events": irrigation_events,
        "capacity": capacity,
    }

    return pd.DataFrame(records), summary


def run_irrigation_experiment(
    n_fields: int = 90,
    days: int = 110,
    seed: int = GLOBAL_SEED,
    example_field_index: int = 0,
    **params: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the irrigation experiment for all policies across many fields.

    The same random seeds are reused across policies so that the comparison is as
    fair as possible: each policy faces the same underlying field conditions and
    stochastic weather process.
    """
    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(1, 10_000_000, size=n_fields)

    summaries: List[Dict[str, float]] = []
    traces: List[pd.DataFrame] = []

    for policy in ["static", "threshold", "bayesian"]:
        for i, field_seed in enumerate(seeds):
            trace, summary = simulate_irrigation_field(policy, int(field_seed), days=days, **params)
            field_id = f"Field_{i + 1:03d}"
            summary.update({"policy": policy, "field_id": field_id})
            summaries.append(summary)

            if i == example_field_index:
                trace = trace.copy()
                trace["field_id"] = field_id
                traces.append(trace)

    return pd.DataFrame(summaries), pd.concat(traces, ignore_index=True)


def run_irrigation_module(output_dir: Path) -> Dict[str, object]:
    """
    Execute subsection 5.2 end-to-end and save all artifacts.
    """
    subdir = output_dir / "5.2"
    subdir.mkdir(parents=True, exist_ok=True)

    summary_df, example_traces = run_irrigation_experiment(**IRRIGATION_PARAMS)

    table_3 = (
        summary_df.groupby("policy")[
            [
                "total_water_mm",
                "stress_days",
                "severe_stress_days",
                "yield_proxy_t_ha",
                "water_productivity",
                "utility_index",
                "irrigation_events",
            ]
        ]
        .mean()
        .reset_index()
        .rename(columns={"policy": "Policy"})
    )
    save_dataframe(table_3, subdir / "Table_3_irrigation_policy_summary.csv")

    # ---- Figure 3: representative field trace.
    static_trace = example_traces[example_traces["policy"] == "static"].copy()
    threshold_trace = example_traces[example_traces["policy"] == "threshold"].copy()
    bayes_trace = example_traces[example_traces["policy"] == "bayesian"].copy()

    plt.figure(figsize=(8.4, 5.6))
    plt.plot(static_trace["day"], static_trace["true_deficit"], label="Static policy deficit")
    plt.plot(threshold_trace["day"], threshold_trace["true_deficit"], label="Threshold policy deficit")
    plt.plot(bayes_trace["day"], bayes_trace["true_deficit"], label="Bayesian policy deficit", linewidth=2.0)
    plt.plot(bayes_trace["day"], bayes_trace["posterior_mean"], linestyle="--", linewidth=1.5, label="Bayesian posterior mean")

    lower = bayes_trace["posterior_mean"] - 1.645 * bayes_trace["posterior_sd"]
    upper = bayes_trace["posterior_mean"] + 1.645 * bayes_trace["posterior_sd"]
    plt.fill_between(bayes_trace["day"], lower, upper, alpha=0.18, label="Bayesian 90% posterior band")

    plt.axhline(float(bayes_trace["stress_threshold"].iloc[0]), linestyle=":", linewidth=1.4, label="Stress threshold")
    irrig_days = bayes_trace.loc[bayes_trace["irrigation_mm"] > 0, "day"].to_numpy()
    irrig_y = np.repeat(float(bayes_trace["stress_threshold"].iloc[0]) + 18.0, len(irrig_days))
    if len(irrig_days) > 0:
        plt.scatter(irrig_days, irrig_y, marker="v", s=28, label="Bayesian irrigation events")

    plt.xlabel("Day of season")
    plt.ylabel("Soil moisture deficit (mm)")
    plt.title("Figure 3. Sequential Bayesian irrigation updating on a representative field")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_3_bayesian_irrigation_trace.png", bbox_inches="tight")
    plt.close()

    # ---- Figure 4: policy trade-off in water use versus stress days.
    plt.figure(figsize=(7.4, 5.4))
    for policy, group in summary_df.groupby("policy"):
        plt.scatter(group["total_water_mm"], group["stress_days"], alpha=0.55, label=policy.title())
    plt.xlabel("Seasonal irrigation water (mm)")
    plt.ylabel("Seasonal stress days")
    plt.title("Figure 4. Water-stress trade-off across irrigation policies")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_4_irrigation_policy_tradeoff.png", bbox_inches="tight")
    plt.close()

    save_dataframe(summary_df, subdir / "irrigation_policy_results_detailed.csv")
    save_dataframe(example_traces, subdir / "irrigation_example_field_trace.csv")

    module_summary = {
        "policy_summary": table_3.round(6).to_dict(orient="records"),
    }
    (subdir / "irrigation_module_summary.json").write_text(json.dumps(module_summary, indent=2), encoding="utf-8")

    return {
        "summary_df": summary_df,
        "example_traces": example_traces,
        "table_3": table_3,
    }


# =============================================================================
# 5) SUBSECTION 5.3 — PEST RISK, INTERVENTION UTILITY, AND VALUE OF INFORMATION
# =============================================================================

PEST_PARAMS = {"sens": 0.74, "spec": 0.79, "second_scout_cost": 2.0}


def bayes_update_binary(prior_p: float, test_positive: bool, sensitivity: float, specificity: float) -> float:
    """
    Apply Bayes' theorem for a binary diagnostic signal.

    The function is used to update outbreak probability from scouting evidence.
    """
    if test_positive:
        numerator = sensitivity * prior_p
        denominator = sensitivity * prior_p + (1.0 - specificity) * (1.0 - prior_p)
    else:
        numerator = (1.0 - sensitivity) * prior_p
        denominator = (1.0 - sensitivity) * prior_p + specificity * (1.0 - prior_p)
    return float(numerator / denominator)


def expected_decision_utility(
    p_outbreak: float,
    spray_cost: float = 32.0,
    environmental_cost: float = 8.0,
    damage_if_no_spray: float = 115.0,
    residual_damage_if_spray: float = 24.0,
) -> Tuple[float, float]:
    """
    Compute expected utility for two actions: spray or wait.

    The utilities are written in cost/loss form. The policy chooses the option
    with the larger expected value (i.e., the less harmful action under the
    current posterior belief).
    """
    eu_spray = -(spray_cost + environmental_cost) - p_outbreak * residual_damage_if_spray
    eu_wait = -p_outbreak * damage_if_no_spray
    return float(eu_spray), float(eu_wait)


def realized_utility_positive(
    outbreak: int,
    spray: bool,
    scouting_cost: float = 0.0,
    spray_cost: float = 32.0,
    environmental_cost: float = 8.0,
    damage_if_no_spray: float = 115.0,
    residual_damage_if_spray: float = 24.0,
    baseline_margin: float = 140.0,
) -> float:
    """
    Convert realized decisions into a positive net-utility index.

    A baseline operating margin is introduced so that the resulting utility index
    is easy to read in tables. Since the baseline is constant across policies, it
    does not affect the ranking of policies.
    """
    utility = baseline_margin - scouting_cost
    if spray:
        utility -= spray_cost + environmental_cost
        if outbreak:
            utility -= residual_damage_if_spray
    else:
        if outbreak:
            utility -= damage_if_no_spray
    return float(utility)


def simulate_pest_dataset(
    n_cases: int = 950,
    seed: int = GLOBAL_SEED,
    sens: float = 0.74,
    spec: float = 0.79,
    second_scout_cost: float = 2.0,
) -> pd.DataFrame:
    """
    Create a synthetic pest-management decision dataset.

    The dataset separates:
        - latent true outbreak probability,
        - a noisier prior model probability,
        - scouting evidence,
        - posterior probabilities after one scout,
        - value of information for an optional second scout.

    This makes it possible to compare simple thresholding against posterior
    decision rules and adaptive information acquisition.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []

    for case_id in range(1, n_cases + 1):
        temp_anomaly = rng.normal(0.0, 1.0)
        humidity = np.clip(rng.normal(0.62, 0.12), 0.25, 0.98)
        leaf_wetness = np.clip(rng.normal(10.5 + 3 * humidity + 1.2 * temp_anomaly, 2.5), 2, 22)
        crop_stage = rng.uniform(0.2, 1.0)
        historical_pressure = np.clip(rng.beta(2.4, 3.0), 0.01, 0.99)
        beneficial_index = np.clip(rng.beta(2.3, 2.1), 0.02, 0.98)
        susceptibility = np.clip(rng.normal(0.0, 0.9), -2, 2)

        # Latent true risk model.
        logit_true = (
            -1.35
            + 0.72 * temp_anomaly
            + 2.0 * (humidity - 0.55)
            + 0.11 * (leaf_wetness - 10)
            + 0.65 * crop_stage
            + 1.25 * (historical_pressure - 0.4)
            - 1.15 * (beneficial_index - 0.45)
            + 0.38 * susceptibility
        )
        p_true = float(expit(logit_true))

        # Noisier prior built from partial information.
        temp_obs = temp_anomaly + rng.normal(0, 0.35)
        humidity_obs = np.clip(humidity + rng.normal(0, 0.06), 0.2, 1.0)
        wet_obs = max(2.0, leaf_wetness + rng.normal(0, 1.8))
        logit_prior = (
            -1.10
            + 0.55 * temp_obs
            + 1.55 * (humidity_obs - 0.55)
            + 0.08 * (wet_obs - 10)
            + 0.40 * crop_stage
            + 0.85 * (historical_pressure - 0.4)
            - 0.75 * (beneficial_index - 0.45)
            + 0.20 * susceptibility
        )
        p_prior = float(np.clip(expit(logit_prior), 0.03, 0.97))

        outbreak = int(rng.random() < p_true)
        scout1_positive = int(rng.random() < (sens if outbreak else (1.0 - spec)))
        p_post1 = bayes_update_binary(p_prior, bool(scout1_positive), sens, spec)

        # EVSI of a second scout.
        p_pos2 = p_post1 * sens + (1.0 - p_post1) * (1.0 - spec)
        p_neg2 = 1.0 - p_pos2
        p_post2_pos = bayes_update_binary(p_post1, True, sens, spec)
        p_post2_neg = bayes_update_binary(p_post1, False, sens, spec)

        eu_spray_1, eu_wait_1 = expected_decision_utility(p_post1)
        current_best_eu = max(eu_spray_1, eu_wait_1)

        eu_after_pos = max(*expected_decision_utility(p_post2_pos))
        eu_after_neg = max(*expected_decision_utility(p_post2_neg))
        evsi_second_scout = float(
            -second_scout_cost + p_pos2 * eu_after_pos + p_neg2 * eu_after_neg - current_best_eu
        )

        rows.append(
            {
                "case_id": case_id,
                "temp_anomaly": temp_anomaly,
                "humidity": humidity,
                "leaf_wetness_hours": leaf_wetness,
                "crop_stage": crop_stage,
                "historical_pressure": historical_pressure,
                "beneficial_index": beneficial_index,
                "susceptibility": susceptibility,
                "p_true": p_true,
                "p_prior": p_prior,
                "outbreak": outbreak,
                "scout1_positive": scout1_positive,
                "p_post1": p_post1,
                "evsi_second_scout": evsi_second_scout,
                "p_post2_pos": p_post2_pos,
                "p_post2_neg": p_post2_neg,
            }
        )

    return pd.DataFrame(rows)


def apply_pest_policies(
    df: pd.DataFrame,
    sens: float = 0.74,
    spec: float = 0.79,
    second_scout_cost: float = 2.0,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Evaluate three decision policies on the simulated pest dataset.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        # ---------------------------------------------------------------------
        # Policy 1: deterministic rule based on the prior probability only.
        # ---------------------------------------------------------------------
        spray_det = bool(row["p_prior"] > 0.55)
        results.append(
            {
                "policy": "Deterministic prior-threshold",
                "case_id": int(row["case_id"]),
                "probability": float(row["p_prior"]),
                "spray": int(spray_det),
                "outbreak": int(row["outbreak"]),
                "utility": realized_utility_positive(int(row["outbreak"]), spray_det),
                "additional_scout": 0,
            }
        )

        # ---------------------------------------------------------------------
        # Policy 2: Bayesian posterior decision after one scouting action.
        # ---------------------------------------------------------------------
        eu_spray, eu_wait = expected_decision_utility(float(row["p_post1"]))
        spray_bayes = bool(eu_spray > eu_wait)
        results.append(
            {
                "policy": "Bayesian posterior decision",
                "case_id": int(row["case_id"]),
                "probability": float(row["p_post1"]),
                "spray": int(spray_bayes),
                "outbreak": int(row["outbreak"]),
                "utility": realized_utility_positive(int(row["outbreak"]), spray_bayes),
                "additional_scout": 0,
            }
        )

        # ---------------------------------------------------------------------
        # Policy 3: Bayesian decision with value-of-information logic.
        # If the second scouting action is expected to be worth its cost,
        # the policy collects new information before acting.
        # ---------------------------------------------------------------------
        use_second = bool(float(row["evsi_second_scout"]) > 0.0)

        if use_second:
            scout2_positive = int(rng.random() < (sens if int(row["outbreak"]) else (1.0 - spec)))
            p_final = float(row["p_post2_pos"]) if scout2_positive else float(row["p_post2_neg"])
            eu_spray_2, eu_wait_2 = expected_decision_utility(p_final)
            spray_voi = bool(eu_spray_2 > eu_wait_2)
            utility_voi = realized_utility_positive(
                int(row["outbreak"]), spray_voi, scouting_cost=second_scout_cost
            )
        else:
            p_final = float(row["p_post1"])
            spray_voi = spray_bayes
            utility_voi = realized_utility_positive(int(row["outbreak"]), spray_voi)

        results.append(
            {
                "policy": "Bayesian + VOI scouting",
                "case_id": int(row["case_id"]),
                "probability": p_final,
                "spray": int(spray_voi),
                "outbreak": int(row["outbreak"]),
                "utility": utility_voi,
                "additional_scout": int(use_second),
            }
        )

    return pd.DataFrame(results)


def calibration_bins(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Build reliability-diagram bins using quantiles.

    Quantile bins are chosen instead of fixed-width bins so that every bin has a
    similar number of cases, which stabilizes the observed outbreak frequency.
    """
    tmp = pd.DataFrame({"prob": probabilities, "outcome": outcomes})
    tmp["bin"] = pd.qcut(tmp["prob"], q=n_bins, duplicates="drop")
    grouped = (
        tmp.groupby("bin", observed=True)
        .agg(mean_prob=("prob", "mean"), observed_rate=("outcome", "mean"), count=("outcome", "size"))
        .reset_index(drop=True)
    )
    return grouped


def pest_policy_summary(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute policy-level decision metrics for subsection 5.3.
    """
    rows = []
    for policy, group in policy_df.groupby("policy"):
        ordered = group.sort_values("case_id")
        probs = ordered["probability"].to_numpy()
        y = ordered["outbreak"].to_numpy()
        spray = ordered["spray"].to_numpy()

        rows.append(
            {
                "Policy": policy,
                "Brier_score": brier_score(y, probs),
                "Spray_rate": float(spray.mean()),
                "Missed_outbreak_rate": float(np.mean((y == 1) & (spray == 0))),
                "Unnecessary_spray_rate": float(np.mean((y == 0) & (spray == 1))),
                "Net_utility_mean": float(ordered["utility"].mean()),
                "Additional_scout_rate": float(ordered["additional_scout"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_pest_module(output_dir: Path) -> Dict[str, object]:
    """
    Execute subsection 5.3 end-to-end and save all artifacts.
    """
    subdir = output_dir / "5.3"
    subdir.mkdir(parents=True, exist_ok=True)

    pest_df = simulate_pest_dataset(**PEST_PARAMS)
    policy_df = apply_pest_policies(pest_df, **PEST_PARAMS)

    table_4 = pest_policy_summary(policy_df)
    save_dataframe(table_4, subdir / "Table_4_pest_policy_summary.csv")

    pest_df = pest_df.copy()
    pest_df["posterior_entropy"] = binary_entropy(pest_df["p_post1"].to_numpy())
    pest_df["uncertainty_bin"] = pd.qcut(pest_df["posterior_entropy"], q=10, duplicates="drop")

    table_5 = (
        pest_df.groupby("uncertainty_bin", observed=True)
        .agg(
            mean_entropy=("posterior_entropy", "mean"),
            mean_evsi=("evsi_second_scout", "mean"),
            positive_evsi_rate=("evsi_second_scout", lambda x: np.mean(x > 0)),
            count=("evsi_second_scout", "size"),
        )
        .reset_index(drop=True)
    )
    save_dataframe(table_5, subdir / "Table_5_evsi_uncertainty_summary.csv")

    # ---- Figure 5: probability calibration.
    prior_cal = calibration_bins(pest_df["p_prior"].to_numpy(), pest_df["outbreak"].to_numpy(), n_bins=10)
    post1_cal = calibration_bins(pest_df["p_post1"].to_numpy(), pest_df["outbreak"].to_numpy(), n_bins=10)
    voi_cases = policy_df[policy_df["policy"] == "Bayesian + VOI scouting"].sort_values("case_id")
    voi_cal = calibration_bins(voi_cases["probability"].to_numpy(), voi_cases["outbreak"].to_numpy(), n_bins=10)

    plt.figure(figsize=(7.2, 5.4))
    plt.plot(prior_cal["mean_prob"], prior_cal["observed_rate"], marker="o", label="Prior risk model")
    plt.plot(post1_cal["mean_prob"], post1_cal["observed_rate"], marker="o", label="Posterior after scouting")
    plt.plot(voi_cal["mean_prob"], voi_cal["observed_rate"], marker="o", label="Final probability with VOI")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Ideal calibration")
    plt.xlabel("Mean predicted outbreak probability")
    plt.ylabel("Observed outbreak rate")
    plt.title("Figure 5. Calibration of pest-risk probabilities")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_5_pest_calibration_comparison.png", bbox_inches="tight")
    plt.close()

    # ---- Figure 6: value of information by uncertainty level.
    plt.figure(figsize=(7.2, 5.4))
    x_positions = np.arange(1, len(table_5) + 1)
    plt.plot(x_positions, table_5["mean_evsi"], marker="o", label="Mean EVSI")
    plt.plot(x_positions, table_5["positive_evsi_rate"], marker="s", label="Positive EVSI rate")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Posterior-uncertainty decile (low to high)")
    plt.ylabel("EVSI / positive-EVSI rate")
    plt.title("Figure 6. Value of additional scouting rises with posterior uncertainty")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(subdir / "Figure_6_pest_evsi_by_uncertainty.png", bbox_inches="tight")
    plt.close()

    save_dataframe(pest_df, subdir / "pest_cases_detailed.csv")
    save_dataframe(policy_df, subdir / "pest_policy_decisions_detailed.csv")

    module_summary = {
        "policy_summary": table_4.round(6).to_dict(orient="records"),
        "uncertainty_summary": table_5.round(6).to_dict(orient="records"),
    }
    (subdir / "pest_module_summary.json").write_text(json.dumps(module_summary, indent=2), encoding="utf-8")

    return {
        "pest_df": pest_df,
        "policy_df": policy_df,
        "table_4": table_4,
        "table_5": table_5,
    }


# =============================================================================
# 6) MASTER ORCHESTRATION
# =============================================================================

def create_zip_archive(source_dir: Path, zip_path: Path) -> None:
    """
    Compress all generated result files into a ZIP archive.
    """
    base_name = zip_path.with_suffix("")
    shutil.make_archive(str(base_name), "zip", root_dir=source_dir)


def write_manifest(output_dir: Path, module_outputs: Dict[str, Dict[str, object]]) -> None:
    """
    Write a compact manifest that summarizes the key quantitative results.

    The manifest is useful when the article text is written later because it
    provides a machine-readable bridge between the computations and the
    manuscript narrative.
    """
    manifest = {}

    yield_table = module_outputs["yield"]["table_1"]
    irrigation_table = module_outputs["irrigation"]["table_3"]
    pest_table = module_outputs["pest"]["table_4"]

    manifest["yield"] = yield_table.round(6).to_dict(orient="records")
    manifest["irrigation"] = irrigation_table.round(6).to_dict(orient="records")
    manifest["pest"] = pest_table.round(6).to_dict(orient="records")

    (output_dir / "results_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """
    Run the entire benchmark and package outputs.
    """
    output_dir = Path.cwd() / "agricultural_bayesian_results"
    ensure_clean_dir(output_dir)

    for name in SUBFOLDERS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    module_outputs = {
        "yield": run_yield_module(output_dir),
        "irrigation": run_irrigation_module(output_dir),
        "pest": run_pest_module(output_dir),
    }

    write_manifest(output_dir, module_outputs)

    zip_path = Path.cwd() / "agricultural_bayesian_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    create_zip_archive(output_dir, zip_path)

    print(f"Results directory: {output_dir}")
    print(f"ZIP archive:       {zip_path}")


if __name__ == "__main__":
    main()
