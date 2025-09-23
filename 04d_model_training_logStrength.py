import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# advanced boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error

from scipy.stats import randint, uniform

import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# Setup
# ---------------------------
img_dir = "04d_model_comparison_plots"
os.makedirs(img_dir, exist_ok=True)

random_state = 42

# ---------------------------
# 1. Caricamento dataset
# ---------------------------
df = pd.read_csv("dataset.csv", sep=";")
df = df.drop(columns=["Unnamed: 0", "id"])
df.columns = df.columns.str.replace("Component", "Comp", regex=True)

# ---------------------------
# 2. Train/test split (con log-transform del target)
# ---------------------------
target = "Strength"
X = df.drop(columns=[target])
y = df[target]

# Trasformazione log (per il training)
y_log = np.log1p(y)

X_trainval, X_test, y_trainval_log, y_test_log = train_test_split(
    X, y_log, test_size=0.15, random_state=random_state
)
X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X_trainval, y_trainval_log, test_size=0.15/0.85, random_state=random_state
)

# Manteniamo i target nello spazio originale per metriche
y_train = np.expm1(y_train_log)
y_valid = np.expm1(y_valid_log)
y_test = np.expm1(y_test_log)
y_trainval = np.expm1(y_trainval_log)

# ---------------------------
# 3. Feature Engineering
# ---------------------------
# def add_engineered_features(df):
#     df = df.copy()
#     df["Binder"] = df[["CementComp", "BlastFurnaceSlag", "FlyAshComp"]].sum(axis=1)
#     df["AggT"] = df[["CoarseAggregateComp", "FineAggregateComp"]].sum(axis=1)
#     df["Tot"] = df[["Binder", "WaterComp", "SuperplasticizerComp", "AggT"]].sum(axis=1)
#     df["W/C"] = df["WaterComp"] / (df["CementComp"] + 1e-6)
#     df["Indicator_Age_7-"] = (df["AgeInDays"] < 7).astype(int)
#     df["Indicator_Age_7_28"] = ((df["AgeInDays"] >= 7) & (df["AgeInDays"] < 28)).astype(int)
#     df["Indicator_Age_28+"] = (df["AgeInDays"] >= 28).astype(int)
#     df = df.drop(columns=["Binder", "Tot", "AggT"])
#     df["Water_Cement"] = df["WaterComp"] * df["CementComp"] 
#     return df

# X_train = add_engineered_features(X_train)
# X_valid = add_engineered_features(X_valid)
# X_test = add_engineered_features(X_test)
# X_trainval = add_engineered_features(X_trainval)P

# ---------------------------
# 4. Preprocessing pipeline
# ---------------------------
num_features = X_train.columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, num_features)]
)

# ---------------------------
# 5. Definizione modelli
# ---------------------------
models = {
    "Ridge": Ridge(alpha=1.0, random_state=random_state),
    "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "BayesianRidge": BayesianRidge(),
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7))]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1))]),
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=random_state),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=random_state),
    "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, verbose=0, random_state=random_state),
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))])
}

# ---------------------------
# 6. Scorers custom (log→exp)
# ---------------------------
def rmse_exp(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    return root_mean_squared_error(y_true, y_pred)

def mae_exp(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_error(y_true, y_pred)

def r2_exp(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    return r2_score(y_true, y_pred)

scoring = {
    "rmse": make_scorer(rmse_exp, greater_is_better=False),
    "mae": make_scorer(mae_exp, greater_is_better=False),
    "r2": make_scorer(r2_exp)
}

cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

# ---------------------------
# 6bis. Valutazione base modelli
# ---------------------------
results = []
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train_log, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({
        "Model": name,
        "RMSE mean": -scores["test_rmse"].mean(),
        "MAE mean": -scores["test_mae"].mean(),
        "R2 mean": scores["test_r2"].mean()
    })

results_df = pd.DataFrame(results).sort_values("RMSE mean")
print(results_df)

# ---------------------------
# 7bis. Hyperparameter tuning
# ---------------------------
top_models = results_df.head(3)["Model"].tolist()
print(f"\nModelli finalisti per tuning: {top_models}")

param_grids = {
    "Ridge": {"model__alpha": uniform(1e-4, 10)},
    "Lasso": {"model__alpha": uniform(1e-4, 1)},
    "ElasticNet": {"model__alpha": uniform(1e-4, 1), "model__l1_ratio": uniform(0, 1)},
    "kNN": {"model__knn__n_neighbors": randint(3, 15), "model__knn__weights": ["uniform", "distance"]},
    "SVR": {"model__svr__C": [1, 10, 100], "model__svr__epsilon": [0.1, 0.2], "model__svr__gamma": ["scale", 0.1, 0.01]},
    "RandomForest": {"model__n_estimators": randint(300, 800), "model__max_depth": [None, 10, 20], "model__max_features": ["sqrt", 0.5]},
    "GradientBoosting": {"model__n_estimators": randint(300, 1000), "model__learning_rate": uniform(0.01, 0.1), "model__max_depth": [3, 5, 7]},
    "MLP": {"model__mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)], "model__mlp__alpha": uniform(1e-5, 1e-2), "model__mlp__learning_rate_init": uniform(1e-4, 1e-2)},
    "XGBoost": {"model__n_estimators": randint(500, 1500), "model__learning_rate": uniform(0.01, 0.1), "model__max_depth": randint(3, 8), "model__subsample": uniform(0.7, 0.3), "model__colsample_bytree": uniform(0.7, 0.3)},
    "CatBoost": {"model__n_estimators": randint(500, 1500), "model__learning_rate": uniform(0.01, 0.1), "model__depth": randint(4, 8)},
}

best_pipelines = {}
summary = []

for name in top_models:
    if name not in param_grids:
        continue
    print(f"========= Hyperparameter tuning of {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[name])])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grids[name],
        n_iter=20,
        cv=cv,
        scoring=make_scorer(rmse_exp, greater_is_better=False),
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train_log)

    print(f"Best params per {name}: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.3f}")

    best_pipelines[name] = search.best_estimator_

    # Valutazione sul validation set (in spazio originale)
    y_val_pred_log = search.best_estimator_.predict(X_valid)
    y_val_pred = np.expm1(y_val_pred_log)
    rmse_val = root_mean_squared_error(y_valid, y_val_pred)

    summary.append({"Model": name, "Validation RMSE": rmse_val})

# Barplot comparativo + selezione modello
if len(summary) > 0:
    summary_df = pd.DataFrame(summary).sort_values("Validation RMSE")
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Model", y="Validation RMSE", data=summary_df)
    plt.title("Prestazioni modelli dopo tuning (Validation RMSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "bplot_rmse_tuning.png"))
    plt.close()

    best_model_name = summary_df.iloc[0]["Model"]
    print(f"\nMiglior modello dopo tuning (validation RMSE): {best_model_name}")
    final_pipe = best_pipelines[best_model_name]
    final_pipe.fit(X_trainval, y_trainval_log)
else:
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nNessun tuning effettuato, uso il miglior modello base: {best_model_name}")
    final_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[best_model_name])])
    final_pipe.fit(X_trainval, y_trainval_log)

# ---------------------------
# 8. Test finale sul best model
# ---------------------------
y_pred_log = final_pipe.predict(X_test)
y_pred = np.expm1(y_pred_log)

rmse_test = root_mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("\n=== Prestazioni su test set ===")
print(f"RMSE = {rmse_test:.3f}")
print(f"MAE  = {mae_test:.3f}")
print(f"R²   = {r2_test:.3f}")

# ---------------------------
# 9. Analisi residui
# ---------------------------
residui = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residui, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title(f"Distribuzione residui - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "residui_hist.png"))
plt.close()
