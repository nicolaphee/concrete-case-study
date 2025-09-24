import sys
sys.path.append("./utils")
from functions import plot_distribution, save_plot, add_engineered_features, correlation_heatmap, plot_univariate_scatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directory di salvataggio immagini
img_dir = "02_eda_plots"
os.makedirs(img_dir, exist_ok=True)

# Carica il dataset
file_path = "dataset.csv"
df_raw = pd.read_csv(file_path, sep=";")

df_raw = df_raw.drop(columns=["Unnamed: 0"])
df_raw = df_raw.drop(columns=["id"])
df_raw.columns = df_raw.columns.str.replace("Component", "Comp", regex=True)

df_dropna = df_raw.dropna()
df_dropna.to_csv("dataset_dropna.csv", sep=";", index=False)
print("Numero campioni, Numero campioni dropna, differenza:")
print(df_raw.shape[0], df_dropna.shape[0], df_raw.shape[0] - df_dropna.shape[0])

df = add_engineered_features(df_raw)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Distribuzioni delle variabili
for col in num_cols:
    plot_distribution(df, col)
    save_plot(f"distribution_{col.replace('/','div')}.png", img_dir=img_dir)
print(f"Grafici di distribuzioni salvate in: {img_dir}")

# Correlazione tra variabili numeriche
correlation_heatmap(df, num_cols)
save_plot("correlation_heatmap.png", img_dir=img_dir)

for col in num_cols:
    plot_univariate_scatter(df, col, "Strength")
    save_plot(f"Strength_vs_{col.replace('/','div')}.png", img_dir=img_dir)
print(f"Grafici di correlazione salvati in: {img_dir}")