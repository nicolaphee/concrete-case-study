import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carica il dataset
file_path = "dataset.csv"
df_raw = pd.read_csv(file_path, sep=";")

# Rimuovo la colonna ridondante 'Unnamed: 0'
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

df_raw = df_raw.drop(columns=["id"])
df_raw.columns = df_raw.columns.str.replace("Component", "Comp", regex=True)

df_dropna = df_raw.dropna()
df_dropna.to_csv("dataset_dropna.csv", sep=";", index=False)
print("Numero campioni, Numero campioni dropna, differenza:")
print(df_raw.shape[0], df_dropna.shape[0], df_raw.shape[0] - df_dropna.shape[0])

df_imputed = df_raw.fillna(df_raw.median())
df_imputed.to_csv("dataset_imputed.csv", sep=";", index=False)

# df = df_raw
df = df_imputed

# Directory di salvataggio immagini
img_dir = "02_eda_plots"
os.makedirs(img_dir, exist_ok=True)

# Visualizzazione distribuzioni con riquadro informativo e salvataggio PNG
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
num_rows = df.shape[0]

for col in num_cols:
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

    plt.tight_layout()
    out_path = os.path.join(img_dir, f"{col}_dist.png")
    plt.savefig(out_path)
    plt.close()

print(f"Tutte le immagini salvate in: {img_dir}")
