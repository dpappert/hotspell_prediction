#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import glob
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss, log_loss
import optuna

torch.set_num_threads(50)

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
# Define MLP model class
# -------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=0, activation='relu'):
        super().__init__()
        layers = []
        if hidden_dim1 > 0:
            layers.append(nn.Linear(input_dim, hidden_dim1))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'linear':
                pass
            if hidden_dim2 > 0:
                layers.append(nn.Linear(hidden_dim1, hidden_dim2))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        else:
            hidden_dim1 = input_dim
        layers.append(nn.Linear(hidden_dim1 if hidden_dim2==0 else hidden_dim2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # No final sigmoid: use BCEWithLogitsLoss
        return self.net(x).squeeze(-1)

# -------------------------------------------------------------------
# Loop over 10 split files (s=1..10)
# -------------------------------------------------------------------

for s in range(1, 11):
    start_time = time.time()
    print(f"\n--- Starting iteration s={s} ---")

    splits = pd.read_csv(path + f"splits_9f_v5/cl{f+1}/k{s}_split_members.txt", sep="\t")
    test_membs = splits.iloc[:, 0]

    full_train_mask = ~TABLE_FULL['member_id'].isin(test_membs)
    X_full_train = TABLE_FULL[full_train_mask].iloc[:, 12:].values
    y_full_train = TABLE_FULL[full_train_mask].iloc[:, 3].values

    global_scaler = StandardScaler()
    global_scaler.fit(X_full_train)

    # -------------------------------------------------------------------
    # Optuna objective for MLP
    # -------------------------------------------------------------------

    def objective(trial):
        hidden_dim1 = trial.suggest_categorical('hidden_dim1', [0, 16, 32, 64])
        hidden_dim2 = trial.suggest_categorical('hidden_dim2', [0, 16, 32])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'linear'])
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-2, log=False)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epochs = 100

        fold_scores = []

        for fold in range(1, 6):
            val_membs = splits.iloc[:, fold]
            train_mask = ~TABLE_FULL['member_id'].isin(pd.concat([val_membs, test_membs]))

            X_train = global_scaler.transform(TABLE_FULL[train_mask].iloc[:, 12:].values)
            y_train = TABLE_FULL[train_mask].iloc[:, 3].values

            X_val = global_scaler.transform(TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 12:].values)
            y_val = TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 3].values

            class_counts = np.bincount(y_train.astype(int))
            class_weights = {i: len(y_train)/class_counts[i] for i in range(len(class_counts))}
            sample_weights = np.array([class_weights[int(y)] for y in y_train])

            train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=50)

            val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=50)

            model = SimpleMLP(X_train.shape[1], hidden_dim1, hidden_dim2, activation)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

            best_val_score = -np.inf
            patience_counter = 0
            for epoch in range(epochs):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(xb)
                    loss = criterion(y_pred, yb)
                    loss.backward()
                    optimizer.step()

                model.eval()
                y_val_pred = []
                with torch.no_grad():
                    for xb, _ in val_loader:
                        y_val_pred.append(model(xb).numpy())
                y_val_pred = np.concatenate(y_val_pred)
                score = compute_normalised_auprc(y_val, y_val_pred)

                scheduler.step(score)

                if score > best_val_score:
                    best_val_score = score
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 10:
                    break

            fold_scores.append(best_val_score)

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
