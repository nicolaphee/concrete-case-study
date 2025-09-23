import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# advanced boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import os

img_dir = "04c_model_comparison_plots"
os.makedirs(img_dir, exist_ok=True)

random_state = 42

# ---------------------------
# 1. Caricamento dataset
# ---------------------------
# Supponiamo che il dataset sia in df con target "Strength"
df = pd.read_csv("dataset.csv", sep=";")
df = df.drop(columns=["Unnamed: 0", "id"])
df.columns = df.columns.str.replace("Component", "Comp", regex=True)

# ---------------------------
# 2. Train/test split
# ---------------------------
target = "Strength"
X = df.drop(columns=[target,])
y = df[target]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=random_state
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, random_state=random_state
)  # 0.15/0.85 = 0.1765 ≈ 15% of total

# ---------------------------
# 3. Feature Engineering
# ---------------------------
apply_feature_eng = True


def add_engineered_features(df):
    df = df.copy()
    df["Binder"] = df[["CementComp", "BlastFurnaceSlag", "FlyAshComp", ]].sum(axis=1)
    df["AggT"] = df[["CoarseAggregateComp", "FineAggregateComp", ]].sum(axis=1)
    df["Tot"] = df[["Binder", "WaterComp", "SuperplasticizerComp", "AggT", ]].sum(axis=1)
    
    df["W/C"] = df["WaterComp"] / (df["CementComp"] + 1e-6)
    # df["W/B"] = df["WaterComp"] / (df["Binder"] + 1e-6)
    # df["(W/B)^2"] = df["W/B"] ** 2

    # df["SP/B"] = df["SuperplasticizerComp"] / (df["Binder"] + 1e-6)
    # df["S%"] = df["BlastFurnaceSlag"] / (df["Binder"] + 1e-6)
    # df["F%"] = df["FlyAshComp"] / (df["Binder"] + 1e-6)
    # df["SCM%"] = (df["BlastFurnaceSlag"] + df["FlyAshComp"]) / (df["Binder"] + 1e-6)
    # df["Binder%"] = df["Binder"] / (df["Tot"] + 1e-6)
    # # df["AggT"]
    # df["AggT/Tot"] = df["AggT"] / (df["Tot"] + 1e-6)
    # df["Sand%"] = df["FineAggregateComp"] / (df["AggT"] + 1e-6)
    # df["Coarse/Fine"] = df["CoarseAggregateComp"] / (df["FineAggregateComp"] + 1e-6)
    # df["AggT/Paste"] = df["AggT"] / (df["Binder"] + df["WaterComp"] + df["SuperplasticizerComp"] + 1e-6)
    # df["logAge"] = np.log1p(df["AgeInDays"])
    # df["sqrtAge"] = np.sqrt(df["AgeInDays"])
    df["Indicator_Age_7-"] = (df["AgeInDays"] < 7).astype(int)
    df["Indicator_Age_7_28"] = ((df["AgeInDays"] >= 7) & (df["AgeInDays"] < 28)).astype(int)
    df["Indicator_Age_28+"] = (df["AgeInDays"] >= 28).astype(int)
    # df["SCM%_logAge"] = df["SCM%"] * df["logAge"]
    # df["SP/B_W/B"] = df["SP/B"] * df["W/B"]
    # df["W/B_Binder%"] = df["W/B"] * df["Binder%"]
    # df["W/B_Sand%"] = df["W/B"] * df["Sand%"]

    df = df.drop(columns=["Binder", "Tot", ])
    df = df.drop(columns=["AggT"])

    # # Interazioni
    df["Water_Cement"] = df["WaterComp"] * df["CementComp"] 
    # df["Water_Superplasticizer"] = df["WaterComp"] * df["SuperplasticizerComp"]
    # df["Age_Cement"] = df["AgeInDays"] * df["CementComp"]
    # df["Age_Superplasticizer"] = df["AgeInDays"] * df["SuperplasticizerComp"]
    # df["Cement_FlyAsh"] = df["CementComp"] * df["FlyAshComp"]
    # df["Cement_BlastFurnaceSlag"] = df["CementComp"] * df["BlastFurnaceSlag"]
    # df["Fine_Coarse"] = df["FineAggregateComp"] * df["CoarseAggregateComp"]

    return df


if apply_feature_eng:
    X_train = add_engineered_features(X_train)
    X_valid = add_engineered_features(X_valid)
    X_test = add_engineered_features(X_test)
    X_trainval = add_engineered_features(X_trainval)

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
    # # Baseline
    # "MeanPredictor": DummyRegressor(strategy="mean"),

    # Baseline lineari
    "Ridge": Ridge(alpha=1.0, random_state=random_state),
    "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "BayesianRidge": BayesianRidge(),

    # Non lineari classici
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7))]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1))]),

    # Tree-based
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=random_state),

    # Boosting avanzati
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=random_state),
    # "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, random_state=random_state),
    "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, verbose=0, random_state=random_state),

    # Rete neurale
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))])
}

# ---------------------------
# 6. Cross-validation
# ---------------------------
scoring = {
    "rmse": make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred, )),
    "mae": make_scorer(mean_absolute_error),
    "r2": make_scorer(r2_score)
}

cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

results = []
all_scores = []  # per distribuzioni metriche

for name, model in models.items():
    print(f"Training and evaluating {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    # Salva medie e deviazioni standard
    results.append({
        "Model": name,
        "RMSE mean": scores["test_rmse"].mean(),
        "RMSE std": scores["test_rmse"].std(),
        "MAE mean": scores["test_mae"].mean(),
        "MAE std": scores["test_mae"].std(),
        "R2 mean": scores["test_r2"].mean(),
        "R2 std": scores["test_r2"].std()
    })
    
    # Salva tutte le fold per grafici
    for i in range(cv.get_n_splits()):
        all_scores.append({
            "Model": name, "Fold": i+1,
            "RMSE": scores["train_rmse"][i],
            "MAE": scores["train_mae"][i],
            "R2": scores["train_r2"][i],
            "Set": "Training"
        })
        all_scores.append({
            "Model": name, "Fold": i+1,
            "RMSE": scores["test_rmse"][i],
            "MAE": scores["test_mae"][i],
            "R2": scores["test_r2"][i],
            "Set": "Validation"
        })


# Tabella riassuntiva
results_df = pd.DataFrame(results).sort_values("RMSE mean")
print(results_df)


# ---------------------------
# 7. Visualizzazione distribuzioni metriche
# ---------------------------
scores_df = pd.DataFrame(all_scores)

# Boxplot per RMSE
plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="RMSE", hue="Set", data=scores_df)
sns.stripplot(x="Model", y="RMSE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
plt.xticks(rotation=45, ha="right")
plt.title("Distribuzione RMSE per modello (CV folds)")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Set")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"boxplot_rmse.png")
plt.savefig(out_path)
plt.close()

# Boxplot per MAE
plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="MAE", hue="Set", data=scores_df)
sns.stripplot(x="Model", y="MAE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
plt.xticks(rotation=45, ha="right")
plt.title("Distribuzione MAE per modello (CV folds)")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Set")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"boxplot_mae.png")
plt.savefig(out_path)
plt.close()

# Boxplot per R²
plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="R2", hue="Set", data=scores_df)
sns.stripplot(x="Model", y="R2", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
plt.xticks(rotation=45, ha="right")
plt.title("Distribuzione R² per modello (CV folds)")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Set")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"boxplot_r2.png")
plt.savefig(out_path)
plt.close()


# ---------------------------
# 7bis. Hyperparameter tuning sui modelli finalisti
# ---------------------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Scegli 2-3 finalisti sulla base del RMSE medio più basso
top_models = results_df.head(3)["Model"].tolist()
print(f"\nModelli finalisti per tuning: {top_models}")

param_grids = {
    "Ridge": {"model__alpha": uniform(1e-4, 10)},
    "Lasso": {"model__alpha": uniform(1e-4, 1)},
    "ElasticNet": {
        "model__alpha": uniform(1e-4, 1),
        "model__l1_ratio": uniform(0, 1),
    },
    "kNN": {
        "model__knn__n_neighbors": randint(3, 15),
        "model__knn__weights": ["uniform", "distance"],
    },
    "SVR": {
        "model__svr__C": [1, 10, 100],
        "model__svr__epsilon": [0.1, 0.2],
        "model__svr__gamma": ["scale", 0.1, 0.01],
    },
    "RandomForest": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", 0.5],
    },
    "GradientBoosting": {
        "model__n_estimators": randint(300, 1000),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [3, 5, 7],
    },
    "MLP": {
        "model__mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "model__mlp__alpha": uniform(1e-5, 1e-2),
        "model__mlp__learning_rate_init": uniform(1e-4, 1e-2),
    },
    "XGBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": randint(3, 8),
        "model__subsample": uniform(0.7, 0.3),        # range 0.7–1.0
        "model__colsample_bytree": uniform(0.7, 0.3), # range 0.7–1.0
    },
    "LightGBM": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [-1, 4, 6, 8],
        "model__num_leaves": randint(31, 128),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "CatBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__depth": randint(4, 8),
    },
}

best_pipelines = {}
for name in top_models:
    print(f"========= Hyperparameter tuning of {name}...")
    if name not in param_grids:
        print(f"Skip tuning per {name} (nessuna griglia definita).")
        continue

    print(f"\nTuning iperparametri per {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[name])])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grids[name],
        n_iter=20,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)

    print(f"Best params per {name}: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.3f}")

    best_pipelines[name] = search.best_estimator_

summary = []
for name, search in best_pipelines.items():
    rmse_cv = -search.score(X_valid, y_valid)  # valutazione sul validation
    summary.append({"Model": name, "Validation RMSE": rmse_cv})

summary_df = pd.DataFrame(summary).sort_values("Validation RMSE")
sns.barplot(x="Model", y="Validation RMSE", data=summary_df)
plt.title("Prestazioni modelli dopo tuning")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"bplot_rmse_tuning.png")
plt.savefig(out_path)
plt.close()

# ---------------------------
# 8. Test finale sul best model (post-tuning se disponibile)
# ---------------------------

if len(best_pipelines) > 0:
    print("\nSelezione modello migliore dai finalisti tunati...")

    # Valuta i finalisti tunati sul validation set
    validation_scores = []
    for name, pipe in best_pipelines.items():
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_valid)
        rmse_val = root_mean_squared_error(y_valid, y_val_pred)
        validation_scores.append((name, rmse_val))
        print(f"{name} - Validation RMSE = {rmse_val:.3f}")

    # Scegli il modello col miglior RMSE
    best_model_name, _ = sorted(validation_scores, key=lambda x: x[1])[0]
    print(f"\nMiglior modello (tuning): {best_model_name}")
    final_pipe = best_pipelines[best_model_name]

    # Retrain su train+valid
    final_pipe.fit(X_trainval, y_trainval)

else:
    print("\nNessun tuning effettuato: uso il miglior modello dalla fase 1.")
    best_model_name = results_df.iloc[0]["Model"]
    final_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[best_model_name])])
    final_pipe.fit(X_trainval, y_trainval)

# Test finale sul set di test
y_pred = final_pipe.predict(X_test)
rmse_test = root_mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("\n=== Prestazioni su test set ===")
print(f"RMSE = {rmse_test:.3f}")
print(f"MAE  = {mae_test:.3f}")
print(f"R²   = {r2_test:.3f}")


# ---------------------------
# 8. Error analysis
# ---------------------------
# Residui
residui = y_test - y_pred

plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residui, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valori predetti")
plt.ylabel("Residui (y_true - y_pred)")
plt.title(f"Residui - {best_model_name}")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"predetti_vs_residui.png")
plt.savefig(out_path)
plt.close()

# Confronto predetti vs osservati
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Osservati (y_true)")
plt.ylabel("Predetti (y_pred)")
plt.title(f"Osservati vs Predetti - {best_model_name}")
# plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"predetti_vs_osservati.png")
plt.savefig(out_path)
plt.close()


# ---------------------------
# 9. Analisi errori avanzata
# ---------------------------

# 1. Distribuzione residui
plt.figure(figsize=(8, 5))
sns.histplot(residui, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residuo (y_true - y_pred)")
plt.ylabel("Conteggio")
plt.title(f"Distribuzione residui - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "residui_hist.png"))
plt.close()

# 2. Residui vs feature chiave (es. AgeInDays, rapporto W/C se presenti)
key_features = []
for feat in ["AgeInDays", "SuperplasticizerComp", "W/C"]:
    if feat in X_test.columns:
        key_features.append(feat)

for feat in key_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test[feat], residui, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(feat)
    plt.ylabel("Residui")
    plt.title(f"Residui vs {feat} - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"residui_vs_{feat.replace('/', 'div')}.png"))
    plt.close()

# 3. Top 10 errori maggiori (in valore assoluto)
errors = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "residuo": residui.abs()
}).sort_values("residuo", ascending=False).head(10)

print("\n=== Top 10 errori maggiori ===")
print(errors)

# Se vuoi salvarli anche su file
errors.to_csv(os.path.join(img_dir, "top10_errori.csv"), index=False)


# ---------------------------
# 10. Feature importance
# ---------------------------
final_model = final_pipe.named_steps["model"]

if hasattr(final_model, "feature_importances_"):
    # Importanza per modelli ad alberi
    feat_names = X_train.columns
    importances = final_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature importance - {best_model_name}")
    # plt.show()
    plt.tight_layout()
    out_path = os.path.join(img_dir, f"feature_importance.png")
    plt.savefig(out_path)
    plt.close()

elif best_model_name in ["Ridge", "Lasso", "ElasticNet", "BayesianRidge"]:
    # Coefficienti per modelli lineari
    feat_names = X_train.columns
    coefs = final_model.coef_
    coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.gca().invert_yaxis()
    plt.title(f"Coefficienti - {best_model_name}")
    # plt.show()
    plt.tight_layout()
    out_path = os.path.join(img_dir, f"feature_importance.png")
    plt.savefig(out_path)
    plt.close()
