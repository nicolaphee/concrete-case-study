import sys
sys.path.append("./utils")  # per importare funzioni da ../utils
from functions import *
from models import models, param_grids

from params import random_state, n_iter, apply_feature_eng, log_transform_target, sample_weighting

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

import joblib

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
plot_performance_metrics(scores_df, results_df, out_prefix="")


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
results_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
print(results_df)

# Visualizzazione distribuzioni delle metriche
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
