import sys
sys.path.append("./utils")  # per importare funzioni da ../utils
from params import img_dir

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

from sklearn.model_selection import RandomizedSearchCV

# crea la cartella per salvare i risultati
os.makedirs(img_dir, exist_ok=True)

#########################
### FUNZIONI GRAFICHE ###
#########################


def save_plot(filename, img_dir):
    plt.tight_layout()
    out_path = os.path.join(img_dir, filename)
    plt.savefig(out_path)
    plt.close()


def plot_distribution(df, col):
    '''
    Plotta la distribuzione di una colonna numerica con istogramma + boxplot + riquadro informativo.
    '''
    
    num_rows = df.shape[0]

    data = df[col].dropna()
    missing = df[col].isnull().sum()
    col_min, col_max = data.min(), data.max()
    mean, median = data.mean(), data.median()
    std = data.std()
    skew = data.skew()
    kurt = data.kurtosis()
    perc5, perc95 = data.quantile(0.05), data.quantile(0.95)

    # Creo la figura
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Distribuzione di {col}")

    # Box con statistiche
    stats_text = (f"Range: {col_min:.2f} - {col_max:.2f}\n"
                f"5°-95° perc: {perc5:.2f} - {perc95:.2f}\n"
                f"Media: {mean:.2f}, Mediana: {median:.2f}\n"
                f"Std: {std:.2f}, Skew: {skew:.2f}, Kurt: {kurt:.2f}\n"
                f"Valori mancanti: {missing} / {num_rows} ({(missing/num_rows)*100:.2f}%)")
    plt.gca().text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.subplot(1,2,2)
    sns.boxplot(x=data)
    plt.title(f"Boxplot di {col}")


def correlation_heatmap(df, num_cols):
    """
    Plotta la matrice di correlazione delle variabili numeriche.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice di correlazione")
    plt.xticks(rotation=25)
    plt.yticks(rotation=55)
    plt.close()


def plot_univariate_scatter(df, col, target):
    """
    Plotta scatterplot tra Strength e una variabile numerica.
    Aggiunge linea di regressione e calcola correlazione Pearson.
    """
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=df[col], y=df[target], alpha=0.6)
    sns.regplot(x=df[col], y=df[target], scatter=False, color="red")
    plt.title(f"{target} vs {col}")

    # Calcolo correlazione Pearson
    corr = df[[col, target]].corr().iloc[0,1]
    plt.gca().text(0.05, 0.95, f"Corr: {corr:.2f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))


def plot_performance_metrics(scores_df, results_df, out_prefix, img_dir):

    # Boxplot per RMSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="RMSE", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="RMSE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione RMSE per modello (CV folds)" if len(out_prefix) == 0 else f"Distribuzione RMSE per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_rmse.png", img_dir)

    # Boxplot per MAE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="MAE", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="MAE", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione MAE per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_mae.png", img_dir)

    # Boxplot per R²
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="R2", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="R2", hue="Set", data=scores_df, palette='dark:black', size=3, jitter=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione R² per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_r2.png", img_dir)

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
    save_plot(f"{out_prefix}overfitting_vs_generalization.png", img_dir)


def plot_final_model_diagnostics(y_test, X_test, final_pipe, best_model_name, img_dir):
    """
    Grafici diagnostici per il modello finale sul test set.
    """
    
    # Residui
    y_pred = final_pipe.predict(X_test)
    residui = y_test - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residui, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valori predetti")
    plt.ylabel("Residui (y_true - y_pred)")
    plt.title(f"Residui - {best_model_name}")
    save_plot("predetti_vs_residui.png", img_dir)

    # Confronto predetti vs osservati
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Osservati (y_true)")
    plt.ylabel("Predetti (y_pred)")
    plt.title(f"Osservati vs Predetti - {best_model_name}")
    save_plot("predetti_vs_osservati.png", img_dir)

    # Distribuzione residui
    plt.figure(figsize=(8, 5))
    sns.histplot(residui, bins=30, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuo (y_true - y_pred)")
    plt.ylabel("Conteggio")
    plt.title(f"Distribuzione residui - {best_model_name}")
    plt.tight_layout()
    save_plot("residui_hist.png", img_dir)

    # Residui vs features
    key_features = []
    for feat in X_test.columns:
        key_features.append(feat)

    for feat in key_features:
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test[feat], residui, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(feat)
        plt.ylabel("Residui")
        plt.title(f"Residui vs {feat} - {best_model_name}")
        plt.tight_layout()
        save_plot(f"residui_vs_{feat.replace('/', 'div')}.png", img_dir)


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
    alpha = 1  # peso per l'overfit gap
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
        if isinstance(model, Pipeline): # when running on 
            pipe = model
        else:
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


###########################
###    MODEL TUNING     ###
###########################

def tune_hyperparameters(
    X_train, y_train, 
    top_models: list,
    models: dict,
    param_grids: dict,
    preprocessor,
    cv: KFold,
    n_iter: int,
    random_state: int,
    sample_weighting: bool = False,
):
    """
    Esegue RandomizedSearchCV sui modelli top selezionati.
    Restituisce i migliori modelli e un sommario delle performance sul validation set.
    """
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

    return best_pipelines


def select_best_tuned_model(
    best_pipelines: dict,
    X_train, y_train,
    X_valid, y_valid,
):
    """
    Valuta i modelli tunati sul validation set e seleziona il migliore.
    """

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

    return final_pipe, best_model_name


def fit_final_model(
    final_pipe,
    X_trainval, y_trainval,
    sample_weighting: bool
):
    if sample_weighting:
        sample_weights_trainval = compute_sample_weights(y_trainval)
        final_fit_params = make_fit_params(final_pipe, sample_weights_trainval)
        final_pipe.fit(X_trainval, y_trainval, **final_fit_params)
    else:
        final_pipe.fit(X_trainval, y_trainval)

    if sample_weighting:
        print("mean weight:", float(sample_weights_trainval.mean()))
        print("supports weights?:", has_fit_parameter(final_pipe.named_steps["model"], "sample_weight"))

    return final_pipe


###########################
###  SHAP EXPLANATION   ###
###########################

def generate_shap_report(final_pipe, X_sample):
    '''
    Genera un report tabellare con osservazioni basate sui valori SHAP.
    '''
    
    import shap

    explainer = shap.Explainer(final_pipe.predict, X_sample)
    shap_values = explainer(X_sample)

    # Genera osservazioni automatiche
    summary = []

    for i, feat in enumerate(X_sample.columns):
        values = shap_values[:, i].values
        mean_abs = np.mean(np.abs(values))
        mean_val = np.mean(values)

        # Classificazione importanza
        if mean_abs >= 2:
            impatto = "Alto impatto"
        elif mean_abs >= 1:
            impatto = "Impatto moderato"
        else:
            impatto = "Impatto debole"

        # Direzione principale
        if mean_val > 0.05:
            direzione = "Valori alti → Strength ↑"
        elif mean_val < -0.05:
            direzione = "Valori alti → Strength ↓"
        else:
            direzione = "Effetto ambiguo / bilanciato"

        # Analisi forma della relazione con quantili
        try:
            quantiles = pd.qcut(X_sample[feat], q=3, duplicates="drop")
            mean_by_bin = pd.DataFrame({"bin": quantiles, "shap": values}).groupby("bin").mean()

            if mean_by_bin["shap"].is_monotonic_increasing:
                forma = "Relazione monotona crescente"
            elif mean_by_bin["shap"].is_monotonic_decreasing:
                forma = "Relazione monotona decrescente"
            else:
                forma = "Relazione non monotona / effetto soglia"
        except Exception:
            forma = "Relazione non stimabile (dati costanti o pochi valori)"

        # Indicazione pratica
        if "↑" in direzione:
            indicazione = f"{impatto}: aumentare {feat} tende ad aumentare la resistenza ({forma})."
        elif "↓" in direzione:
            indicazione = f"{impatto}: valori elevati di {feat} tendono a ridurre la resistenza ({forma})."
        else:
            indicazione = f"{impatto}: {feat} non mostra una direzione chiara ({forma})."

        summary.append({
            "Feature": feat,
            "Importanza media SHAP": round(mean_abs, 2),
            "Osservazione": direzione,
            "Relazione stimata": forma,
            "Indicazione pratica": indicazione
        })

    df_summary = pd.DataFrame(summary).sort_values("Importanza media SHAP", ascending=False)

    return df_summary, shap_values
