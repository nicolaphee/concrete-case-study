import matplotlib.pyplot as plt
import seaborn as sns

import os

from adjustText import adjust_text

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

import numpy as np
import pandas as pd

from sklearn.utils.validation import has_fit_parameter

from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import train_test_split, cross_validate, KFold

from sklearn.pipeline import Pipeline

# crea la cartella per salvare i risultati
img_dir = "07_results"
os.makedirs(img_dir, exist_ok=True)

#########################
### FUNZIONI GRAFICHE ###
#########################


def save_plot(filename, img_dir=img_dir):
    plt.tight_layout()
    out_path = os.path.join(img_dir, filename)
    plt.savefig(out_path)
    plt.close()


def plot_performance_metrics(scores_df, results_df, out_prefix):

    # Boxplot per RMSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="RMSE", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="RMSE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione RMSE per modello (CV folds)" if len(out_prefix) == 0 else f"Distribuzione RMSE per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_rmse.png")

    # Boxplot per MAE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="MAE", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="MAE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione MAE per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_mae.png")

    # Boxplot per R²
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="R2", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="R2", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione R² per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_r2.png")

    # Scatter plot RMSE train vs RMSE validation con barre di errore
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        results_df["RMSE mean train"], results_df["RMSE mean"],  # x=train, y=val
        xerr=results_df["RMSE std train"], 
        yerr=results_df["RMSE std"],
        fmt="o", capsize=5, label="Modelli"
    )
    # aggiungi label ai punti
    texts = []
    for i, row in results_df.iterrows():
        texts.append(
            plt.text(
                row["RMSE mean train"],
                row["RMSE mean"],
                row["Model"],
                fontsize=9
            )
        )
    # riposiziona le label per evitare overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    # linea di bilanciamento perfetto
    min_val = min(results_df["RMSE mean train"].min(), results_df["RMSE mean"].min())
    max_val = max(results_df["RMSE mean train"].max(), results_df["RMSE mean"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray", label="Perfetto bilanciamento")
    plt.xlabel("RMSE Train (mean ± std)")
    plt.ylabel("RMSE Validation (mean ± std)")
    plt.title(f"Overfitting vs Generalizzazione (con deviazione standard) - {out_prefix}")
    plt.legend()
    save_plot(f"{out_prefix}overfitting_vs_generalization.png")


##########################
###  SAMPLE WEIGHTING  ###
##########################

# calcola pesi campione basati sui valori del target
def compute_sample_weights(y):
    # first try
    # # più peso ai valori alti del target
    # w = np.log1p(y)        # crescita dolce, evita esplosioni
    # return w / w.mean()    # normalizza attorno a 1

    # second try
    # peso proporzionale al quadrato del target
    w = (y / y.mean())**2
    return w


def make_fit_params(pipe, sample_weights):
    """
    Ritorna un dict di fit_params per cross_validate.
    Passa sample_weight solo se l'estimatore finale lo supporta.
    Gestisce sia TransformedTargetRegressor che modelli "puri".
    """
    model = pipe.named_steps["model"]

    # Caso 1: modello wrappato (TransformedTargetRegressor)
    if isinstance(model, TransformedTargetRegressor):
        reg = model.regressor
        if has_fit_parameter(reg, "sample_weight"):
            return {"model__sample_weight": sample_weights}
        else:
            return {}

    # Caso 2: modello normale
    else:
        if has_fit_parameter(model, "sample_weight"):
            return {"model__sample_weight": sample_weights}
        else:
            return {}
        
##########################
###  TARGET TRANSFORM  ###
##########################
# log1p per ridurre squilibrio sugli estremi
log_transformer = FunctionTransformer(np.log1p, np.expm1, validate=True)


def wrap_with_target_transformer(model):
    return TransformedTargetRegressor(regressor=model, transformer=log_transformer)


#############################
### FEATURES ENGINEERING  ###
#############################

def add_engineered_features(df):
    df = df.copy()
    # df["Binder"] = df[["CementComp", "BlastFurnaceSlag", "FlyAshComp", ]].sum(axis=1)
    # df["AggT"] = df[["CoarseAggregateComp", "FineAggregateComp", ]].sum(axis=1)
    # df["Tot"] = df[["Binder", "WaterComp", "SuperplasticizerComp", "AggT", ]].sum(axis=1)
    
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
    # df["Indicator_Age_7-"] = (df["AgeInDays"] < 7).astype(int)
    # df["Indicator_Age_7_28"] = ((df["AgeInDays"] >= 7) & (df["AgeInDays"] < 28)).astype(int)
    # df["Indicator_Age_28+"] = (df["AgeInDays"] >= 28).astype(int)
    # df["SCM%_logAge"] = df["SCM%"] * df["logAge"]
    # df["SP/B_W/B"] = df["SP/B"] * df["W/B"]
    # df["W/B_Binder%"] = df["W/B"] * df["Binder%"]
    # df["W/B_Sand%"] = df["W/B"] * df["Sand%"]

    # df = df.drop(columns=["Binder", "Tot", ])
    # df = df.drop(columns=["AggT"])

    # # Interazioni
    # df["Water_Cement"] = df["WaterComp"] * df["CementComp"] 
    # df["Water_Superplasticizer"] = df["WaterComp"] * df["SuperplasticizerComp"]
    # df["Age_Cement"] = df["AgeInDays"] * df["CementComp"]
    # df["Age_Superplasticizer"] = df["AgeInDays"] * df["SuperplasticizerComp"]
    # df["Cement_FlyAsh"] = df["CementComp"] * df["FlyAshComp"]
    # df["Cement_BlastFurnaceSlag"] = df["CementComp"] * df["BlastFurnaceSlag"]
    # df["Fine_Coarse"] = df["FineAggregateComp"] * df["CoarseAggregateComp"]

    return df



###########################
###   MODEL SELECTION   ###
###########################

def composite_score(scores):
    alpha = 0.5  # peso per l'overfit gap
    beta = 0  # peso per la deviazione standard 
    train_rmse_mean = scores["train_rmse"].mean()
    test_rmse_mean = scores["test_rmse"].mean()
    test_rmse_std = scores["test_rmse"].std()
    overfit_gap = test_rmse_mean - train_rmse_mean

    return test_rmse_mean + alpha * overfit_gap + beta * test_rmse_std


def cross_validate_models(
    X_train, y_train,
    models: dict,
    cv: KFold,
    preprocessor,
    composite_score,
    sample_weighting: bool = False,
):
    """
    Esegue cross-validation sui modelli forniti, calcolando metriche di performance e overfitting.
    """
    scoring = {
        "rmse": make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred, )),
        "mae": make_scorer(mean_absolute_error),
        "r2": make_scorer(r2_score)
    }

    results = []
    all_scores = []  # per distribuzioni metriche

    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        if sample_weighting:
            # calcola pesi sul target train
            sample_weights = compute_sample_weights(y_train)
            fit_params = make_fit_params(pipe, sample_weights)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True, params=fit_params)
        else:
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        # Calcola composite score (per bilanciare accuratezza e overfitting)
        train_rmse_mean = scores["train_rmse"].mean()
        train_rmse_std = scores["train_rmse"].std()
        test_rmse_mean = scores["test_rmse"].mean()
        test_rmse_std = scores["test_rmse"].std()

        # Salva medie e deviazioni standard
        results.append({
            "Model": name,
            "MAE mean": scores["test_mae"].mean(),
            "MAE std": scores["test_mae"].std(),
            "R2 mean": scores["test_r2"].mean(),
            "R2 std": scores["test_r2"].std(),
            "RMSE mean": test_rmse_mean,
            "RMSE std": test_rmse_std,
            "RMSE mean train": train_rmse_mean,
            "RMSE std train": train_rmse_std,
            "Composite score": composite_score(scores),
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
    # results_df = pd.DataFrame(results).sort_values("RMSE mean")
    results_df = pd.DataFrame(results).sort_values("Composite score")
    scores_df = pd.DataFrame(all_scores)

    return results_df, scores_df