import sys
sys.path.append("./utils")  # per importare funzioni da ../utils
from functions import *

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# advanced boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

import joblib

# # per ignorare warning
# import warnings
# warnings.filterwarnings("ignore")

random_state = 42

n_iter = 20  # numero di iterazioni per RandomizedSearchCV
apply_feature_eng = True
log_transform_target = True  # se True, applica log-transform al target
sample_weighting = True  # se True, usa pesi campione
if sample_weighting:
    print("Usiamo pesi campione basati sui valori del target.")


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
    # "Ridge": Ridge(alpha=1.0, random_state=random_state),
    "Ridge": Ridge(alpha=10.0, random_state=random_state),
    # "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
    "Lasso": Lasso(alpha=0.05, random_state=random_state, max_iter=10000),
    # "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "BayesianRidge": BayesianRidge(),

    # Non lineari classici
    # "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7))]),
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance"))]),
    # "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1))]),
    # "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=1, epsilon=0.2))]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=0.1, epsilon=0.3))]),

    # Tree-based
    # "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state),
    # "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5, random_state=random_state),
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, random_state=random_state),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, min_samples_split=7, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, learning_rate=0.05, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=7, learning_rate=0.05, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=7, min_samples_split=7, learning_rate=0.05, random_state=random_state),

    # Boosting avanzati
    # "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=random_state),
    # "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.025, max_depth=5, random_state=random_state),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=random_state),
    # "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, verbose=0, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.025, depth=5, early_stopping_rounds=100, verbose=0, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.01, depth=4, early_stopping_rounds=100, verbose=0, random_state=random_state),
    "CatBoost": CatBoostRegressor(n_estimators=500, learning_rate=0.03, depth=3, early_stopping_rounds=100, l2_leaf_reg=3, verbose=0, random_state=random_state),

    # Rete neurale
    # "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))])
    # "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, alpha=0.001, early_stopping=True, validation_fraction=0.2, random_state=random_state))]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, alpha=0.01, early_stopping=True, validation_fraction=0.2, random_state=random_state))]),

    # PLS: riduce dimensionalità → minore rischio di overfit con molte variabili collineari
    "PLSRegression": PLSRegression(n_components=5),  # 5 componenti è un buon default

    # Regressori robusti: resistenti a outlier e rumore
    "Huber": Pipeline([("scaler", RobustScaler()),("huber", HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=10000))]),
    "RANSAC": Pipeline([("scaler", RobustScaler()),("ransac", RANSACRegressor(min_samples=0.5, max_trials=100, random_state=random_state))]),
    "TheilSen": Pipeline([("scaler", RobustScaler()),("theilsen", TheilSenRegressor(max_subpopulation=10000, random_state=random_state))]),

    # # Gaussian Process con kernel RBF regolarizzato
    # "GaussianProcess": Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("gpr", GaussianProcessRegressor(
    #         kernel=C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0),
    #         alpha=1.0,          # regolarizzazione (rumore) relativamente alta
    #         normalize_y=True,
    #         random_state=random_state
    #     ))
    # ]),

    # # Bagging con regressori lineari regolarizzati
    # "BaggingRidge": BaggingRegressor(
    #     estimator=Pipeline([("scaler", StandardScaler()),("ridge", Ridge(alpha=1.0, random_state=random_state))]),
    #     n_estimators=50,
    #     max_samples=0.8,
    #     max_features=0.8,
    #     random_state=random_state
    # ),
    # "BaggingLasso": BaggingRegressor(
    #     estimator=Pipeline([("scaler", StandardScaler()),("lasso", Lasso(alpha=0.01, random_state=random_state, max_iter=10000))]),
    #     n_estimators=50,
    #     max_samples=0.8,
    #     max_features=0.8,
    #     random_state=random_state
    # ),

}
if log_transform_target:
    print("Applichiamo log-transform al target.")
    models = {name: wrap_with_target_transformer(model) for name, model in models.items()}

# ---------------------------
# 6. Cross-validation - Prima valutazione modelli
# ---------------------------
results_df, scores_df = cross_validate_models(
    X_train, y_train,
    models,
    preprocessor,
    random_state,
    composite_score=composite_score,
    sample_weighting=sample_weighting,
)
results_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_df)

# Visualizzazione distribuzioni delle metriche
plot_performance_metrics(scores_df, results_df, out_prefix="")


# ---------------------------
# 7bis. Hyperparameter tuning sui modelli finalisti
# ---------------------------
param_grids = {
    "Ridge": {"model__alpha": uniform(1e-4, 10)},
    "Lasso": {"model__alpha": uniform(1e-4, 1)},
    "ElasticNet": {
        "model__alpha": uniform(1e-4, 1),
        "model__l1_ratio": uniform(0, 1),
    },
    "kNN": {
        "model__knn__n_neighbors": randint(5, 30),
        "model__knn__weights": ["uniform", "distance"],
    },
    "SVR": {
        "model__svr__C": [0.1, 1, 10],
        "model__svr__epsilon": [0.1, 0.2, 0.3],
        "model__svr__gamma": ["scale", 0.1, 0.01],
    },
    "RandomForest": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [5, 10, 15],
        "model__max_features": ["sqrt", 0.5],
        "model__min_samples_leaf": randint(2, 10),
    },
    "ExtraTrees": {
        "model__n_estimators": randint(300, 800),      # numero alberi
        "model__max_depth": [5, 10, 15],         # None = senza limite
        "model__max_features": ["sqrt", 0.5, 0.7],     # meno feature = più robusto
        "model__min_samples_leaf": randint(2, 10),     # ↑ min_leaf = meno overfit
        "model__min_samples_split": randint(2, 10),    # ↑ split threshold
    },
    "GradientBoosting": {
        "model__n_estimators": randint(300, 1000),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [2, 3, 5],
        "model__subsample": uniform(0.7, 0.3),
    },
    "XGBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": randint(3, 6),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "LightGBM": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [-1, 4, 6],
        "model__num_leaves": randint(15, 64),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "CatBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__depth": randint(3, 6),
        "model__l2_leaf_reg": uniform(1, 10),
    },
    "MLP": {
        "model__mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "model__mlp__alpha": uniform(1e-4, 1e-2),
        "model__mlp__learning_rate_init": uniform(1e-4, 1e-2),
        "model__mlp__early_stopping": [True],
    },
    "PLSRegression": {
        "model__n_components": randint(2, 10),
    },
    "Huber": {
        "model__huber__alpha": uniform(1e-4, 1.0),
        "model__huber__epsilon": uniform(1.2, 0.8),  # 1.2–2.0
    },
    "RANSAC": {
        "model__ransac__min_samples": uniform(0.4, 0.5),  # 0.4–0.9
        "model__ransac__max_trials": randint(50, 200),
        "model__ransac__residual_threshold": uniform(1.0, 5.0),
    },
    "TheilSen": {
        "model__theilsen__max_subpopulation": [1000, 5000, 10000],
        "model__theilsen__tol": uniform(1e-4, 1e-2),
    },
    # "GaussianProcess": {
    #     "model__gpr__alpha": uniform(1e-2, 10.0),
    #     "model__gpr__kernel": [
    #         C(1.0, (1e-2, 1e2)) * RBF(length_scale=l)
    #         for l in [0.5, 1.0, 2.0, 5.0]
    #     ],
    # },
    # "BaggingRidge": {
    #     "model__estimator__ridge__alpha": uniform(1e-3, 10.0),
    #     "model__n_estimators": randint(20, 100),
    #     "model__max_samples": uniform(0.5, 0.5),   # 0.5–1.0
    #     "model__max_features": uniform(0.5, 0.5),
    # },
    # "BaggingLasso": {
    #     "model__estimator__lasso__alpha": uniform(1e-3, 1.0),
    #     "model__n_estimators": randint(20, 100),
    #     "model__max_samples": uniform(0.5, 0.5),
    #     "model__max_features": uniform(0.5, 0.5),
    # },
}
if log_transform_target:
    # aggiusta i nomi dei parametri per il target transformer
    param_grids = {
        name: {key.replace("model__", "model__regressor__"): value for key, value in grid.items()}
        for name, grid in param_grids.items()
    }

# Scegli 2-3 finalisti
top_models = results_df.sort_values("Composite score").head(3)["Model"].tolist()
print(f"\nModelli finalisti per tuning: {top_models}")

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
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
    )
    if sample_weighting:
        sample_weights_train = compute_sample_weights(y_train)
        fit_params_search = make_fit_params(pipe, sample_weights_train)
        search.fit(X_train, y_train, **fit_params_search)
    else:
        search.fit(X_train, y_train)

    print(f"Best params per {name}: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.3f}")

    best_pipelines[name] = search.best_estimator_

summary = []
for name, search in best_pipelines.items():
    # # R^2 selection
    # rmse_cv = -search.score(X_valid, y_valid)  # valutazione sul validation
    # summary.append({"Model": name, "Validation R^2": rmse_cv})

    # RMSE selection
    y_pred_val = search.predict(X_valid)
    rmse_val = root_mean_squared_error(y_valid, y_pred_val)
    summary.append({"Model": name, "Validation RMSE": rmse_val})


summary_df = pd.DataFrame(summary).sort_values("Validation RMSE")
sns.barplot(x="Model", y="Validation RMSE", data=summary_df)
plt.title("Prestazioni modelli dopo tuning")
save_plot("tuning_summary.png")


# ---------------------------
# 7ter. Comparazione modelli tunati (train vs validation)
# ---------------------------

scoring = {
    "rmse": make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)),
    "mae": make_scorer(mean_absolute_error),
    "r2": make_scorer(r2_score)
}

results_tuned = []
all_scores_tuned = []

for name, pipeline in best_pipelines.items():
    print(f"Valutazione finale del modello tunato: {name}")
    if sample_weighting:
        # calcola pesi sul target train
        sample_weights = compute_sample_weights(y_train)
        fit_params = make_fit_params(pipeline, sample_weights)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True, params=fit_params)
    else:
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    # Calcola composite score (per bilanciare accuratezza e overfitting)
    train_rmse_mean = scores["train_rmse"].mean()
    train_rmse_std = scores["train_rmse"].std()
    test_rmse_mean = scores["test_rmse"].mean()
    test_rmse_std = scores["test_rmse"].std()
    # overfit_gap = test_rmse_mean - train_rmse_mean
    # composite_score = test_rmse_mean + alpha * overfit_gap + beta * test_rmse_std

    # Salva medie e deviazioni standard
    results_tuned.append({
        "Model": name,
        "MAE mean": scores["test_mae"].mean(),
        "MAE std": scores["test_mae"].std(),
        "R2 mean": scores["test_r2"].mean(),
        "R2 std": scores["test_r2"].std(),
        "RMSE mean": test_rmse_mean,
        "RMSE std": test_rmse_std,
        "RMSE mean train": train_rmse_mean,
        "RMSE std train": train_rmse_std,
        # "Overfit gap": overfit_gap,
        # "Composite score": composite_score,
    })
    
    # Tutte le fold (train + validation)
    for i in range(cv.get_n_splits()):
        all_scores_tuned.append({
            "Model": name, "Fold": i+1,
            "RMSE": scores["train_rmse"][i],
            "MAE": scores["train_mae"][i],
            "R2": scores["train_r2"][i],
            "Set": "Training"
        })
        all_scores_tuned.append({
            "Model": name, "Fold": i+1,
            "RMSE": scores["test_rmse"][i],
            "MAE": scores["test_mae"][i],
            "R2": scores["test_r2"][i],
            "Set": "Validation"
        })

# Tabella riassuntiva
results_tuned_df = pd.DataFrame(results_tuned).sort_values("RMSE mean")
results_tuned_df.to_csv(os.path.join(img_dir, "tuned_models_summary.csv"), index=False)
print(results_tuned_df)

# ---------------------------
# Visualizzazione distribuzioni metriche (train vs validation)
# ---------------------------
scores_tuned_df = pd.DataFrame(all_scores_tuned)
plot_performance_metrics(scores_tuned_df, results_tuned_df, out_prefix="tuned_")

# ---------------------------
# 8. Selezione finale sul best model
# ---------------------------

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

# Retrain su train+validation
print("Retraining finale sul train+validation set...")
if sample_weighting:
    sample_weights_trainval = compute_sample_weights(y_trainval)
    final_fit_params = make_fit_params(final_pipe, sample_weights_trainval)
    final_pipe.fit(X_trainval, y_trainval, **final_fit_params)
else:
    final_pipe.fit(X_trainval, y_trainval)

if sample_weighting:
    print("mean weight:", float(sample_weights_trainval.mean()))
    print("supports weights?:", has_fit_parameter(final_pipe.named_steps["model"], "sample_weight"))


# Test finale sul set di test
y_pred = final_pipe.predict(X_test)
rmse_test = root_mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
mae_test_rel = mae_test / (y_test.max() - y_test.min())

print("\n=== Prestazioni su test set ===")
print(f"RMSE = {rmse_test:.3f}")
print(f"MAE  = {mae_test:.3f}")
print(f"R²   = {r2_test:.3f}")
print(f"MAE relativo = {mae_test_rel:.3f}")


# ---------------------------
# 9. Error analysis
# ---------------------------
# Residui
residui = y_test - y_pred

plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residui, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valori predetti")
plt.ylabel("Residui (y_true - y_pred)")
plt.title(f"Residui - {best_model_name}")
save_plot("predetti_vs_residui.png")

# Confronto predetti vs osservati
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Osservati (y_true)")
plt.ylabel("Predetti (y_pred)")
plt.title(f"Osservati vs Predetti - {best_model_name}")
save_plot("predetti_vs_osservati.png")

# Distribuzione residui
plt.figure(figsize=(8, 5))
sns.histplot(residui, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residuo (y_true - y_pred)")
plt.ylabel("Conteggio")
plt.title(f"Distribuzione residui - {best_model_name}")
plt.tight_layout()
save_plot("residui_hist.png")

# Residui vs feature chiave (es. AgeInDays, rapporto W/C se presenti)
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
    save_plot(f"residui_vs_{feat.replace('/', 'div')}.png")

# Top 10 errori maggiori (in valore assoluto)
errors = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "residuo": residui.abs()
}).sort_values("residuo", ascending=False).head(10)

print("\n=== Top 10 errori maggiori ===")
print(errors)
errors.to_csv(os.path.join(img_dir, "top10_errori.csv"), index=False)

# Salva il modello finale
final_model = final_pipe.named_steps["model"]
joblib.dump(final_pipe, os.path.join(img_dir, "final_model.pkl"))

# # ---------------------------
# # 10. Feature importance
# # ---------------------------
# final_model = final_pipe.named_steps["model"]
# loaded_pipe = joblib.load(os.path.join("06_results", "final_model.pkl"))

# if hasattr(final_model, "feature_importances_"):
#     # Importanza per modelli ad alberi
#     feat_names = X_train.columns
#     importances = final_model.feature_importances_
#     imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)

#     plt.figure(figsize=(8, 6))
#     plt.barh(imp_df["Feature"], imp_df["Importance"])
#     plt.gca().invert_yaxis()
#     plt.title(f"Feature importance - {best_model_name}")
#     save_plot("feature_importance.png")

# elif best_model_name in ["Ridge", "Lasso", "ElasticNet", "BayesianRidge"]:
#     # Coefficienti per modelli lineari
#     feat_names = X_train.columns
#     coefs = final_model.coef_
#     coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)

#     plt.figure(figsize=(8, 6))
#     plt.barh(coef_df["Feature"], coef_df["Coefficient"])
#     plt.gca().invert_yaxis()
#     plt.title(f"Coefficienti - {best_model_name}")
#     save_plot("feature_importance.png")


# # ---------------------------
# # 11. Interpretabilità del modello
# # ---------------------------
# import shap
# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.tree import DecisionTreeRegressor, export_text

# # ======================
# # 3. SHAP analysis
# # ======================
# explainer = shap.Explainer(final_model, X_train)
# shap_values = explainer(X_train)

# # Importanza globale
# shap.plots.bar(shap_values, max_display=15)
# save_plot("shap_global_importance.png")

# # ======================
# # 4. Partial Dependence
# # ======================
# features_to_plot = ["CementComp", "WaterComp", "AgeInDays"]
# PartialDependenceDisplay.from_estimator(final_model, X_train, 
#                                         features=features_to_plot, 
#                                         grid_resolution=50)
# save_plot("partial_dependence.png")

# # ======================
# # 5. Surrogate model per estrarre regole
# # ======================
# # Usiamo un albero decisionale che approssima le predizioni del modello principale
# y_pred_train = final_model.predict(X_train)

# surrogate = DecisionTreeRegressor(max_depth=3, random_state=random_state)
# surrogate.fit(X_train, y_pred_train)

# # Stampa le regole in formato leggibile
# rules = export_text(surrogate, feature_names=list(X_train.columns))
# print("\nRegole surrogate (da interpretare come linee guida):\n")
# print(rules)
# with open(os.path.join(img_dir, "surrogate_rules.txt"), "w") as f:
#     f.write(rules)


# # ======================
# # 1. Predizioni
# # ======================
# y_pred = final_model.predict(X_train)

# # ======================
# # 2. Definizione fasce Strength
# # ======================
# # Dividiamo Strength previsto in 3 fasce (terzili)
# bins = pd.qcut(y_pred, q=3, labels=["Bassa", "Media", "Alta"])
# df_ranges = X_train.copy()
# df_ranges["Strength_pred"] = y_pred
# df_ranges["Classe_Strength"] = bins

# # ======================
# # 3. Calcolo range per fascia
# # ======================
# summary = {}
# for classe in ["Bassa", "Media", "Alta"]:
#     subset = df_ranges[df_ranges["Classe_Strength"] == classe].drop(columns=["Strength_pred", "Classe_Strength"])
#     summary[classe] = pd.DataFrame({
#         "Min": subset.quantile(0.05),
#         "Mediana": subset.median(),
#         "Max": subset.quantile(0.95)
#     })

# # Combiniamo in un’unica tabella
# tables = []
# for classe, tab in summary.items():
#     tab["Classe_Strength"] = classe
#     tables.append(tab.reset_index().rename(columns={"index": "Feature"}))

# final_table = pd.concat(tables, axis=0).reset_index(drop=True)

# # ======================
# # 4. Visualizzazione
# # ======================
# import tabulate
# print(tabulate.tabulate(final_table, headers="keys", tablefmt="pretty"))



# # ======================
# # 1. Boxplot per ogni feature
# # ======================
# for col in X_train.columns:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x="Classe_Strength", y=col, data=df_ranges,
#                 order=["Bassa", "Media", "Alta"],
#                 palette="Set2")
#     plt.title(f"Distribuzione di {col} per fasce di Strength prevista")
#     plt.xlabel("Classe Strength")
#     plt.ylabel(col)
#     save_plot(f"boxplot_{col.replace('/', 'div')}_by_strength_class.png")

# # ======================
# # 2. Heatmap dei range
# # ======================
# # Creiamo una matrice pivot con Min/Max per visualizzazione
# heatmap_data = final_table.pivot(index="Feature", 
#                                  columns="Classe_Strength", 
#                                  values="Mediana")

# plt.figure(figsize=(8,6))
# sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f")
# plt.title("Mediana delle feature per fascia di Strength")
# plt.ylabel("Feature")
# plt.xlabel("Classe Strength")
# save_plot("heatmap_feature_ranges.png")
