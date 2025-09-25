import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

######################################
#### FUNZIONI PER CLASSIFICAZIONE  ###
######################################

def composite_score_classification(scores):
    """
    Combina performance e robustezza in classificazione.
    Usa l'F1 macro medio (da massimizzare) e penalizza la deviazione standard.
    """
    f1_mean = scores["test_f1_macro"].mean()
    f1_std = scores["test_f1_macro"].std()
    accuracy_mean = scores["test_accuracy"].mean()

    # Formula: più alto è meglio → usiamo negativo per compatibilità con sort ascending
    return -(0.7 * f1_mean + 0.3 * accuracy_mean - 0.1 * f1_std)


def cross_validate_models_classification(
    X_train, y_train,
    models: dict,
    cv: KFold,
    preprocessor,
    composite_score_fn,
):
    """
    Esegue cross-validation sui modelli di classificazione, calcolando accuracy e F1 macro.
    """
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
    }

    results = []
    all_scores = []

    for name, model in models.items():
        print(f"Training and evaluating {name}...")

        if isinstance(model, Pipeline) and "scaler" not in model.named_steps:
            pipe = model
        else:
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)

        # Salva medie e deviazioni standard
        results.append({
            "Model": name,
            "Accuracy mean": scores["test_accuracy"].mean(),
            "Accuracy std": scores["test_accuracy"].std(),
            "F1_macro mean": scores["test_f1_macro"].mean(),
            "F1_macro std": scores["test_f1_macro"].std(),
            "Composite score": composite_score_fn(scores),
        })

        # Salva tutte le fold
        for i in range(cv.get_n_splits()):
            all_scores.append({
                "Model": name, "Fold": i+1,
                "Accuracy": scores["train_accuracy"][i],
                "F1_macro": scores["train_f1_macro"][i],
                "Set": "Training"
            })
            all_scores.append({
                "Model": name, "Fold": i+1,
                "Accuracy": scores["test_accuracy"][i],
                "F1_macro": scores["test_f1_macro"][i],
                "Set": "Validation"
            })

    results_df = pd.DataFrame(results).sort_values("Composite score")
    scores_df = pd.DataFrame(all_scores)

    return results_df, scores_df


#########################
### FUNZIONI GRAFICHE ###
#########################
from adjustText import adjust_text
import matplotlib.pyplot as plt
import seaborn as sns
from functions import save_plot

def plot_performance_metrics_classification(scores_df, results_df, out_prefix, img_dir):
    """
    Visualizza distribuzione e trade-off delle metriche di classificazione
    (Accuracy, F1 macro) per modelli e cross-validation.
    """
    # Boxplot Accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Accuracy", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="Accuracy", hue="Set", data=scores_df, 
                  palette='dark:black', size=3, jitter=True, dodge=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione Accuracy per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_accuracy.png", img_dir)

    # Boxplot F1 macro
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="F1_macro", hue="Set", data=scores_df)
    sns.stripplot(x="Model", y="F1_macro", hue="Set", data=scores_df, 
                  palette='dark:black', size=3, jitter=True, dodge=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuzione F1 macro per modello (CV folds) - {out_prefix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Set")
    save_plot(f"{out_prefix}boxplot_f1macro.png", img_dir)

    # Scatter plot Accuracy train vs Validation
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        results_df["Accuracy mean"], results_df["F1_macro mean"], 
        xerr=results_df["Accuracy std"], yerr=results_df["F1_macro std"],
        fmt="o", capsize=5, label="Modelli"
    )
    texts = []
    for i, row in results_df.iterrows():
        texts.append(
            plt.text(
                row["Accuracy mean"],
                row["F1_macro mean"],
                row["Model"],
                fontsize=9
            )
        )
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    plt.xlabel("Accuracy (mean ± std)")
    plt.ylabel("F1 macro (mean ± std)")
    plt.title(f"Accuracy vs F1 macro (con deviazione standard) - {out_prefix}")
    plt.legend()
    save_plot(f"{out_prefix}accuracy_vs_f1macro.png", img_dir)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_confusion_matrices_train_valid(y_train, y_train_pred, 
                                        y_valid, y_valid_pred, 
                                        class_names, 
                                        img_dir, normalize=True, 
                                        ):
    """
    Plotta le matrici di confusione per train e validation affiancate.
    
    Args:
        y_train, y_train_pred: etichette vere e predette sul train
        y_valid, y_valid_pred: etichette vere e predette sul validation
        class_names: lista con i nomi delle classi (nell'ordine corretto)
        normalize: se True mostra percentuali, altrimenti conteggi grezzi
        img_dir: directory per il salvataggio
        filename: nome file immagine
    """
    # Train
    cm_train = confusion_matrix(y_train, y_train_pred, labels=class_names)
    if normalize:
        cm_train = cm_train.astype("float") / cm_train.sum(axis=1)[:, np.newaxis]

    # Validation
    cm_valid = confusion_matrix(y_valid, y_valid_pred, labels=class_names)
    if normalize:
        cm_valid = cm_valid.astype("float") / cm_valid.sum(axis=1)[:, np.newaxis]

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_train, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Matrice di confusione - Train")
    axes[0].set_xlabel("Predetto")
    axes[0].set_ylabel("Vero")

    sns.heatmap(cm_valid, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Matrice di confusione - Validation")
    axes[1].set_xlabel("Predetto")
    axes[1].set_ylabel("Vero")

    save_plot("confusion_matrix.png", img_dir)

def tune_hyperparameters_classification(
    X_train, y_train, 
    top_models: list,
    models: dict,
    param_grids: dict,
    preprocessor,
    cv,
    n_iter: int,
    random_state: int,
):
    """
    Esegue RandomizedSearchCV sui modelli di classificazione top selezionati.
    Restituisce i migliori modelli e un sommario delle performance sul validation set.
    """
    best_pipelines = {}
    for name in top_models:
        print(f"========= Hyperparameter tuning of {name}...")
        if name not in param_grids:
            print(f"Skip tuning per {name} (nessuna griglia definita).")
            continue

        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[name])])

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",   # 🔑 usiamo F1 macro per classi bilanciate
            n_jobs=-1,
            random_state=random_state,
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        print(f"Best params per {name}: {search.best_params_}")
        print(f"Best CV F1-macro: {search.best_score_:.3f}")

        best_pipelines[name] = search.best_estimator_

    return best_pipelines


def select_best_tuned_model_classification(
    best_pipelines: dict,
    X_train, y_train,
    X_valid, y_valid,
):
    """
    Valuta i modelli tunati sul validation set e seleziona il migliore.
    Usa F1 macro come criterio principale.
    """
    validation_scores = []
    for name, pipe in best_pipelines.items():
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_valid)
        f1_val = f1_score(y_valid, y_val_pred, average="macro")
        validation_scores.append((name, f1_val))
        print(f"{name} - Validation F1-macro = {f1_val:.3f}")

    # Scegli il modello col miglior F1 macro
    best_model_name, _ = sorted(validation_scores, key=lambda x: x[1], reverse=True)[0]
    print(f"\nMiglior modello (tuning): {best_model_name}")
    final_pipe = best_pipelines[best_model_name]

    return final_pipe, best_model_name


def fit_final_model_classification(
    final_pipe,
    X_trainval, y_trainval,
):
    """
    Allena il modello finale di classificazione su train+validation.
    """
    final_pipe.fit(X_trainval, y_trainval)
    print("Final model trained on train+validation set.")
    return final_pipe


def evaluate_final_model_classification(
    final_pipe,
    X_trainval, y_trainval,
    X_test, y_test,
    class_names,
    img_dir,
):
    """
    Valuta il modello finale di classificazione su train+validation e test.
    Stampa classification_report, confusion_matrix e salva i risultati.
    Genera anche i plot grafici delle confusion matrix.
    """
    results = {}

    # --- Train+Validation ---
    y_trainval_pred = final_pipe.predict(X_trainval)
    print("\n=== Performance su Train+Validation ===")
    print(classification_report(y_trainval, y_trainval_pred, target_names=class_names))

    cm_train = confusion_matrix(y_trainval, y_trainval_pred, labels=[0,1,2])
    print("Matrice di confusione (Train+Validation):")
    print(pd.DataFrame(cm_train, index=class_names, columns=class_names))

    results["trainval_report"] = classification_report(
        y_trainval, y_trainval_pred, target_names=class_names, output_dict=True
    )

    # --- Test ---
    y_test_pred = final_pipe.predict(X_test)
    print("\n=== Performance su Test ===")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    cm_test = confusion_matrix(y_test, y_test_pred, labels=[0,1,2])
    print("Matrice di confusione (Test):")
    print(pd.DataFrame(cm_test, index=class_names, columns=class_names))

    results["test_report"] = classification_report(
        y_test, y_test_pred, target_names=class_names, output_dict=True
    )

    # --- Salvataggio CSV ---
    df_trainval = pd.DataFrame(results["trainval_report"]).T
    df_test = pd.DataFrame(results["test_report"]).T
    df_trainval.to_csv(os.path.join(img_dir, "final_performance_trainval.csv"))
    df_test.to_csv(os.path.join(img_dir, "final_performance_test.csv"))

    # --- Plot grafici ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Matrice di confusione - Train+Validation")
    axes[0].set_xlabel("Predetto")
    axes[0].set_ylabel("Vero")

    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Matrice di confusione - Test")
    axes[1].set_xlabel("Predetto")
    axes[1].set_ylabel("Vero")

    save_plot("final_confusion_matrices.png", img_dir)

    return results