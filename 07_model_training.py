import sys
sys.path.append("./utils")  # per importare funzioni da ../utils
from functions import plot_performance_metrics, plot_final_model_diagnostics
from functions import add_engineered_features, define_imputer_preprocessor, wrap_with_target_transformer
from functions import composite_score, cross_validate_models, tune_hyperparameters, select_best_tuned_model, fit_final_model

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from models import models, param_grids

from params import random_state, n_iter, apply_feature_eng, log_transform_target, sample_weighting

import joblib

import tabulate

img_dir = "07_results"
os.makedirs(img_dir, exist_ok=True)

# # per ignorare warning
# import warnings
# warnings.filterwarnings("ignore")


# ---------------------------
# 1. Caricamento dataset
# ---------------------------
# Supponiamo che il dataset sia in df con target "Strength"
df = pd.read_csv("dataset.csv", sep=";")
df = df.drop(columns=["Unnamed: 0", "id"])
df.columns = df.columns.str.replace("Component", "Comp", regex=True)

# ---------------------------
# 2. Feature Engineering
# ---------------------------
target = "Strength"
X = df.drop(columns=[target,])
y = df[target]

if apply_feature_eng:
    X = add_engineered_features(X)
    # X = X.drop(columns=[])

# ---------------------------
# 3. Train/test split
# ---------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=random_state
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, random_state=random_state
)  # 0.15/0.85 = 0.1765 ≈ 15% of total

# ---------------------------
# 4. Preprocessing pipeline
# ---------------------------
num_features = X_train.columns
preprocessor = define_imputer_preprocessor(num_features, random_state)

# ---------------------------
# 5. Definizione modelli
# ---------------------------

if log_transform_target:
    print("Applichiamo log-transform al target.")
    models = {name: wrap_with_target_transformer(model) for name, model in models.items()}

if sample_weighting:
    print("Usiamo pesi campione basati sui valori del target quando necessario.")

# ---------------------------
# 6. Cross-validation - Prima valutazione modelli
# ---------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

results_df, scores_df = cross_validate_models(
    X_train, y_train,
    models,
    cv=cv,
    preprocessor=preprocessor,
    composite_score=composite_score,
    sample_weighting=sample_weighting,
)
results_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_df)

# Visualizzazione distribuzioni delle metriche
plot_performance_metrics(scores_df, results_df, out_prefix="", img_dir=img_dir)


# ---------------------------
# 7bis. Hyperparameter tuning sui modelli finalisti
# ---------------------------
if log_transform_target:
    # aggiusta i nomi dei parametri per il target transformer
    param_grids = {
        name: {key.replace("model__", "model__regressor__"): value for key, value in grid.items()}
        for name, grid in param_grids.items()
    }

# Scegli 2-3 finalisti
top_models = results_df.sort_values("Composite score").head(3)["Model"].tolist()
print(f"\nModelli finalisti per tuning: {top_models}")

best_pipelines = tune_hyperparameters(
    X_train, y_train, 
    top_models=top_models,
    models=models,
    param_grids=param_grids,
    preprocessor=preprocessor,
    cv=cv,
    n_iter=n_iter,
    random_state=random_state,
    sample_weighting=sample_weighting
)

# ---------------------------
# 7ter. Comparazione modelli tunati (train vs validation)
# ---------------------------
results_tuned_df, scores_tuned_df = cross_validate_models(
    X_train, y_train,
    best_pipelines,
    cv=cv,
    preprocessor=preprocessor,
    composite_score=composite_score,
    sample_weighting=sample_weighting,
)
results_tuned_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_tuned_df)

# Visualizzazione distribuzioni delle metriche
plot_performance_metrics(scores_tuned_df, results_tuned_df, out_prefix="tuned_", img_dir=img_dir)


# ---------------------------
# 8. Selezione finale sul best model
# ---------------------------

print("\nSelezione modello migliore dai finalisti tunati...")
final_pipe, best_model_name = select_best_tuned_model(
    best_pipelines,
    X_train, y_train,
    X_valid, y_valid,
)

# Retrain su train+validation
print("Retraining finale sul train+validation set...")
final_pipe = fit_final_model(
    final_pipe,
    X_trainval, y_trainval,
    sample_weighting=sample_weighting
)

# Valutazione finale sul set di test
y_test_pred = final_pipe.predict(X_test)
rmse_test = root_mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_test_rel = mae_test / (y_test.max() - y_test.min())

# Valutazione finale sul set di train+validation
y_trainval_pred = final_pipe.predict(X_trainval)
rmse_trainval = root_mean_squared_error(y_trainval, y_trainval_pred)
mae_trainval = mean_absolute_error(y_trainval, y_trainval_pred)
r2_trainval = r2_score(y_trainval, y_trainval_pred)
mae_trainval_rel = mae_trainval / (y_trainval.max() - y_trainval.min())

final_performance_df = pd.DataFrame({
    "Set": ["train+valid", "test", ],
    "RMSE": [rmse_trainval, rmse_test, ],
    "MAE": [mae_trainval, mae_test, ],
    "R2": [r2_trainval, r2_test, ],
    "MAE relativo": [mae_trainval_rel, mae_test_rel,],
})
print(tabulate.tabulate(final_performance_df, headers="keys", tablefmt="pretty"))

final_performance_df.to_csv(os.path.join(img_dir, "final_performance.csv"), index=False)

# ---------------------------
# 9. Error analysis
# ---------------------------
plot_final_model_diagnostics(y_test, X_test, final_pipe, best_model_name, img_dir)

# Top 10 errori maggiori (in valore assoluto)
residui = y_test - y_test_pred
errors = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_test_pred,
    "residuo": residui.abs()
}).sort_values("residuo", ascending=False).head(10)

print("\n=== Top 10 errori maggiori ===")
print(errors)
errors.to_csv(os.path.join(img_dir, "top10_errori.csv"), index=False)

# Salva il modello finale
joblib.dump(final_pipe, os.path.join(img_dir, "final_pipeline.pkl"))
