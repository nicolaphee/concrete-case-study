import sys
sys.path.append("./utils")
import os 
from params import img_dir, random_state
img_dir = os.path.join(img_dir, "08_explain_model")
os.makedirs(img_dir, exist_ok=True)

from functions import add_engineered_features, save_plot, generate_shap_report

import joblib

import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

# # per ignorare warning
# import warnings
# warnings.filterwarnings("ignore")


#######################
###  IMPORTA INPUT  ###
#######################

# Carica modello finale
final_pipeline = joblib.load(os.path.join("07_results", "final_pipeline.pkl"))
# print(final_pipeline)

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

# Campiona un sottoinsieme dei dati
X_sample = X.sample(n=min(200, len(X)), random_state=random_state)


#########################
###  SHAP EXPLAINER   ###
#########################
df_shap_summary, shap_values = generate_shap_report(final_pipeline, X_sample)
print(df_shap_summary)
df_shap_summary.to_excel(os.path.join(img_dir, "shap_feature_report.xlsx"), index=False)
print(f"Report generato: {os.path.join(img_dir, 'shap_feature_report.xlsx')}")

# Bar plot: importanza media assoluta delle feature
shap.plots.bar(shap_values, max_display=15)
plt.title("SHAP Feature importance (mean absolute SHAP value)")
save_plot("bar_plot.png", img_dir=img_dir)

# Beeswarm plot: effetti direzionali delle feature
shap.plots.beeswarm(shap_values, max_display=15)
plt.title("SHAP Feature effects (beeswarm)")
save_plot("beeswarm_plot.png", img_dir=img_dir)


# Dependence plot su features ingegnerizzata (esempio: W/C)
for feat in X.columns:
    shap.plots.scatter(shap_values[:, feat], color=shap_values)
    plt.title(f"SHAP Feature effects (dependence) - {feat}")
    save_plot(f"dependence_plot_{feat.replace('/', 'div')}.png", img_dir=img_dir)