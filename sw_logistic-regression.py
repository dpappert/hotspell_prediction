#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import glob
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    brier_score_loss, log_loss
)
from sklearn.inspection import permutation_importance
import optuna

# -------------------------------------------------------------------
# Load initial data
# -------------------------------------------------------------------

path = '/scratch3/dpappert/CESM2/ML/'
folders = ['targets_cl1_HWs_m50_18512000_fmstat_1.5sd_r1',
           'targets_cl2_HWs_m50_18512000_fmstat_1.5sd_r1',
           'targets_cl4_HWs_m50_18512000_fmstat_1.5sd_r1']

f = 0
TABLE_init = pd.read_csv(path + f'RF/targets/{folders[f]}.txt', sep='\t')
member_ids = np.unique(TABLE_init['member_id'])

file_list = sorted(glob.glob(f"{path}/RF/preds/{folders[f]}/*"))
preds_ = [pd.read_csv(file, sep='\t') for file in file_list]

TABLE_FULL = pd.concat([TABLE_init, pd.concat(preds_, axis=1)], axis=1)

# Remove duration==7 and add binary label
TABLE_FULL = TABLE_FULL[TABLE_FULL['DURATION'] != 7]
TABLE_FULL.insert(3, 'DURATION_bin', (TABLE_FULL['DURATION'] > 7).astype(int))

# -------------------------------------------------------------------
# Metric: Normalised AUPRC
# -------------------------------------------------------------------

def compute_normalised_auprc(y_true, y_score):
    baseline = y_true.mean()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = auc(recall, precision)
    if baseline == 1.0:
        return 1.0
    return (auc_pr - baseline) / (1 - baseline)

# -------------------------------------------------------------------
# Loop over 10 split files (s=1..10)
# -------------------------------------------------------------------

for s in range(1, 11):
    start_time = time.time()
    print(f"\n--- Starting iteration s={s} ---")

    splits = pd.read_csv(path + f"splits_9f_v5/cl{f+1}/k{s}_split_members.txt", sep="\t")
    test_membs = splits.iloc[:, 0]

    # Define full_train mask (used for global scaler)
    full_train_mask = ~TABLE_FULL['member_id'].isin(test_membs)

    X_full_train = TABLE_FULL[full_train_mask].iloc[:, 12:]
    y_full_train = TABLE_FULL[full_train_mask].iloc[:, 3]

    # Fit one scaler on full_train
    global_scaler = StandardScaler()
    global_scaler.fit(X_full_train)

    # -------------------------------------------------------------------
    # Optuna objective
    # -------------------------------------------------------------------

    def objective(trial):

        C = trial.suggest_float("C", 1e-4, 10, log=True)
        penalty = trial.suggest_categorical("penalty", ["l2"])

        solver = "lbfgs"
        max_iter = 2000

        fold_scores = []

        for fold in range(1, 6):

            val_membs = splits.iloc[:, fold]

            train_mask = ~TABLE_FULL['member_id'].isin(
                pd.concat([val_membs, test_membs])
            )

            X_train = TABLE_FULL[train_mask].iloc[:, 12:]
            y_train = TABLE_FULL[train_mask].iloc[:, 3]

            X_val = TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 12:]
            y_val = TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 3]

            # Scale using global full_train scaler
            X_train_scaled = global_scaler.transform(X_train)
            X_val_scaled   = global_scaler.transform(X_val)

            model = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                class_weight="balanced",
                max_iter=max_iter,
                n_jobs=20
            )

            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]

            fold_scores.append(compute_normalised_auprc(y_val, y_proba))

        mean_norm = np.mean(fold_scores)
        std_norm = np.std(fold_scores)

        print(f"Trial {trial.number} | mean={mean_norm:.4f} | std={std_norm:.4f}")

        return mean_norm - std_norm

    # -------------------------------------------------------------------
    # Run Optuna study
    # -------------------------------------------------------------------

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    best_params = study.best_trial.params

    print(f"\nBest trial for s={s}: {study.best_trial.value:.4f}")
    print("Params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------
    # Retrain final model on full training set (scaled)
    # -------------------------------------------------------------------

    X_train_full_scaled = global_scaler.transform(X_full_train)

    X_test = TABLE_FULL[TABLE_FULL['member_id'].isin(test_membs)].iloc[:, 12:]
    y_test = TABLE_FULL[TABLE_FULL['member_id'].isin(test_membs)].iloc[:, 3]
    X_test_scaled = global_scaler.transform(X_test)

    final_model = LogisticRegression(
        **best_params,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        n_jobs=20
    )

    final_model.fit(X_train_full_scaled, y_full_train)
    y_test_proba = final_model.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------

    roc_auc = roc_auc_score(y_test, y_test_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)
    baseline = y_test.mean()
    norm_auprc = (pr_auc - baseline) / (1 - baseline)
    brier = brier_score_loss(y_test, y_test_proba)
    logloss = log_loss(y_test, y_test_proba)
    baseline_logloss = log_loss(y_test, np.full_like(y_test, baseline, dtype=float))

    metrics_df = pd.DataFrame({
        "roc_auc": [round(roc_auc, 4)],
        "pr_auc": [round(pr_auc, 4)],
        "baseline": [round(baseline, 4)],
        "norm_auprc": [round(norm_auprc, 4)],
        "brier_score": [round(brier, 4)],
        "log_loss": [round(logloss, 4)],
        "baseline_logloss": [round(baseline_logloss, 4)]
    })
    metrics_df.to_csv(path + f"LR/scores/{folders[f]}/k{s}_test_metrics.csv", index=False)

    # Save test predictions for curves
    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred_proba": y_test_proba})
    preds_df.to_csv(path + f"LR/scores/{folders[f]}/k{s}_test_predictions.csv", index=False)

    # -------------------------------------------------------------------
    # Permutation Importance (top 20)
    # -------------------------------------------------------------------

    result = permutation_importance(
        final_model,
        X_test_scaled,
        y_test,
        scoring='average_precision',
        n_repeats=10,
        random_state=42,
        n_jobs=20
    )

    feature_importances = pd.DataFrame({
        "feature_name": X_test.columns,
        "importance_value": result.importances_mean
    }).sort_values("importance_value", ascending=False).head(20)

    feature_importances.to_csv(
        path + f"LR/scores/{folders[f]}/k{s}_feature_importance.csv", index=False
    )

    # -------------------------------------------------------------------
    # Time tracking
    # -------------------------------------------------------------------
    duration = (time.time() - start_time) / 60
    print(f"--- Iteration s={s} completed in {duration:.2f} minutes ---")
