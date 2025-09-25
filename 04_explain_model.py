import os 
from utils.params import random_state
img_dir = "04_explain_model"
os.makedirs(img_dir, exist_ok=True)

multivariable_dependence = False

from utils.functions import add_engineered_features, drop_excluded_columns, save_plot
from utils.functions import generate_shap_report, generate_optimal_scenarios

import joblib

import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

# per ignorare warning
import warnings
warnings.filterwarnings("ignore")


#######################
###  IMPORTA INPUT  ###
#######################

# Carica modello finale
final_pipeline = joblib.load(os.path.join("03_results", "final_pipeline.pkl"))

# Carica dataset
df = pd.read_csv("dataset.csv", sep=";")
df = df.drop(columns=["Unnamed: 0", "id"])
df.columns = df.columns.str.replace("Component", "Comp", regex=True)


#########################
###   PREPROCESSING   ###
#########################
# Definisci target e features
target = "Strength"
X = df.drop(columns=[target,])
y = df[target]

# Aggiungi feature ingegnerizzate
X = add_engineered_features(X)
X = drop_excluded_columns(X)

# Campiona un sottoinsieme dei dati
X_sample = X.sample(n=min(200, len(X)), random_state=random_state)


#########################
###  SHAP EXPLAINER   ###
#########################
df_shap_summary, shap_values = generate_shap_report(final_pipeline, X_sample, window=10, img_dir=img_dir)
print(df_shap_summary)
df_shap_summary.to_excel(os.path.join(img_dir, "shap_feature_report.xlsx"), index=False)
print(f"Report generato: {os.path.join(img_dir, 'shap_feature_report.xlsx')}")

# Bar plot: importanza media assoluta delle feature
shap.plots.bar(shap_values, max_display=15)
plt.title("SHAP Feature importance (mean absolute SHAP value)")
save_plot("bar_plot.png", img_dir=img_dir)

# Beeswarm plot: effetti direzionali delle feature
shap.plots.beeswarm(shap_values, max_display=15)
plt.title("SHAP Feature effects (beeswarm<>)")
save_plot("beeswarm_plot.png", img_dir=img_dir)


# Dependence plot su features
for feat in X.columns:
    if multivariable_dependence:
        shap.plots.scatter(shap_values[:, feat], color=shap_values)
    else:
        shap.plots.scatter(shap_values[:, feat], color=shap_values[:, feat])
    plt.title(f"SHAP Feature effects (dependence) - {feat}")
    save_plot(f"dependence_plot_{feat.replace('/', 'div')}.png", img_dir=img_dir)

optimal_ranges = {
   "AgeInDays_cat": [lambda rng: rng.integers(1, 3)],
   "Binder": [lambda rng: rng.uniform(238.1, 374.0)],
   "W/C": [lambda rng: rng.uniform(0.3, 0.51)],
   "AggT/Paste": [lambda rng: rng.uniform(2.19, 3.15)],
   "SuperplasticizerComp": [lambda rng: rng.uniform(3.9, 20.0)],
   "CementComp": [lambda rng: rng.uniform(300, 500)],
}
df_best = generate_optimal_scenarios(final_pipeline, optimal_ranges, n_samples=1000, top_k=5, img_dir=img_dir)