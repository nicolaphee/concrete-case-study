from utils.functions import save_plot
from utils.functions_classifers import cross_validate_models_classification, composite_score_classification, plot_performance_metrics_classification
from utils.functions_classifers import tune_hyperparameters_classification, select_best_tuned_model_classification, fit_final_model_classification, evaluate_final_model_classification

from utils.functions import plot_performance_metrics, plot_final_model_diagnostics
from utils.functions import add_engineered_features, define_imputer_preprocessor, wrap_with_target_transformer
from utils.functions import composite_score, cross_validate_models, tune_hyperparameters, select_best_tuned_model, fit_final_model

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from classifier_models import models, param_grids

from params import random_state, n_iter, apply_feature_eng, log_transform_target, sample_weighting, use_simple_imputer

import joblib

import tabulate

from sklearn.preprocessing import LabelEncoder


img_dir = "09_results"
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


# Categorizzazione Strength in tre classi
low_thr = 25 #np.percentile(y, 33)
high_thr = 45 #np.percentile(y, 66)

def categorize_strength(val):
    if val <= low_thr:
        return "Bassa"
    elif val <= high_thr:
        return "Media"
    else:
        return "Alta"

y = y.apply(categorize_strength)

# Encode in numeri (0,1,2) per XGBoost & Co.
le = LabelEncoder()
y = le.fit_transform(y)


pd.DataFrame(y).hist()
save_plot("target_distribution_categorized.png", img_dir=img_dir)



if apply_feature_eng:
    X = add_engineered_features(X)
    # X = X.drop(columns=[
    # # "CementComp",
    # "WaterComp",
    # "BlastFurnaceSlag",
    # "FlyAshComp",
    # "SuperplasticizerComp",
    # # "CoarseAggregateComp",
    # "FineAggregateComp",
    # # "AgeInDays",
    # ])

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
preprocessor = define_imputer_preprocessor(num_features, random_state, use_simple_imputer)

# # ---------------------------
# # 5. Definizione modelli
# # ---------------------------

# if log_transform_target:
#     print("Applichiamo log-transform al target.")
#     models = {name: wrap_with_target_transformer(model) for name, model in models.items()}

# if sample_weighting:
#     print("Usiamo pesi campione basati sui valori del target quando necessario.")

# ---------------------------
# 6. Cross-validation - Prima valutazione modelli
# ---------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

results_df, scores_df = cross_validate_models_classification(
    X_train, y_train,
    models,
    cv=cv,
    preprocessor=preprocessor,
    composite_score_fn=composite_score_classification,
)

results_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_df)

# Visualizzazione distribuzioni delle metriche
plot_performance_metrics_classification(scores_df, results_df, out_prefix="", img_dir=img_dir)


# plot_confusion_matrices_train_valid(y_train, y_train_pred, y_valid, y_valid_pred, class_names, img_dir, normalize=True, )


# ---------------------------
# 7bis. Hyperparameter tuning sui modelli finalisti
# ---------------------------
# if log_transform_target:
#     # aggiusta i nomi dei parametri per il target transformer
#     param_grids = {
#         name: {key.replace("model__", "model__regressor__"): value for key, value in grid.items()}
#         for name, grid in param_grids.items()
#     }

# Scegli 2-3 finalisti
top_models = results_df.sort_values("F1_macro mean").head(3)["Model"].tolist()
print(f"\nModelli finalisti per tuning: {top_models}")

best_pipelines = tune_hyperparameters_classification(
    X_train, y_train, 
    top_models,
    models,
    param_grids,
    preprocessor,
    cv,
    n_iter,
    random_state,
)

# ---------------------------
# 7ter. Comparazione modelli tunati (train vs validation)
# ---------------------------
results_tuned_df, scores_tuned_df = cross_validate_models_classification(
    X_train, y_train,
    best_pipelines,
    cv=cv,
    preprocessor=preprocessor,
    composite_score_fn=composite_score_classification,
)

results_tuned_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_tuned_df)

# Visualizzazione distribuzioni delle metriche
plot_performance_metrics_classification(scores_tuned_df, results_tuned_df, out_prefix="tuned_", img_dir=img_dir)


# ---------------------------
# 8. Selezione finale sul best model
# ---------------------------

print("\nSelezione modello migliore dai finalisti tunati...")
final_pipe, best_model_name = select_best_tuned_model_classification(
    best_pipelines,
    X_train, y_train,
    X_valid, y_valid,
)

# Retrain su train+validation
print("Retraining finale sul train+validation set...")
final_pipe = fit_final_model_classification(
    final_pipe,
    X_trainval, y_trainval,
)

results = evaluate_final_model_classification(
    final_pipe,
    X_trainval, y_trainval,
    X_test, y_test,
    class_names=['Bassa', 'Media', 'Alta'],   # se hai usato LabelEncoder
    img_dir=img_dir,
)