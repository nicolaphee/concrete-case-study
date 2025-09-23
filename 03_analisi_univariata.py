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

df_imputed = df_raw.fillna(df_raw.median())

df = df_imputed

# Directory di salvataggio immagini
img_dir = "03_eda_plots_univariate"
os.makedirs(img_dir, exist_ok=True)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
# Heatmap di correlazione con etichette ruotate
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di correlazione")
plt.xticks(rotation=25)
plt.yticks(rotation=55)
heatmap_path = os.path.join(img_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Variabili numeriche predittive
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Strength")

# Analisi univariata: scatterplot Strength vs ciascun predittore
for col in num_cols:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=df[col], y=df["Strength"], alpha=0.6)
    sns.regplot(x=df[col], y=df["Strength"], scatter=False, color="red")
    plt.title(f"Strength vs {col}")

    # Calcolo correlazione Pearson
    corr = df[[col, "Strength"]].corr().iloc[0,1]
    plt.gca().text(0.05, 0.95, f"Corr: {corr:.2f}",
                   transform=plt.gca().transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    out_path = os.path.join(img_dir, f"Strength_vs_{col}.png")
    plt.savefig(out_path)
    plt.close()

print(f"Grafici univariati salvati in: {img_dir}")
