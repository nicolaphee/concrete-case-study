import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Carica il dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path, sep=";")

# Rimuovo la colonna ridondante 'Unnamed: 0'
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Rimuovo righe con valori mancanti
df = df.dropna()

# Variabili predittive e target
X = df.drop(columns=["Strength", "id"])  # tolgo anche 'id' perché non predittivo
y = df["Strength"]

# Aggiungo costante per regressione OLS
X = sm.add_constant(X)

# Regressione lineare multipla
model = sm.OLS(y, X).fit()
print(model.summary())

# Directory di salvataggio immagini
img_dir = "eda_plots_multivariate"
os.makedirs(img_dir, exist_ok=True)

# Residuals vs Predicted
plt.figure(figsize=(7,5))
predicted = model.fittedvalues
residuals = model.resid
sns.scatterplot(x=predicted, y=residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Strength")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "residuals_vs_predicted.png"))
plt.close()

# Distribuzione dei residui
plt.figure(figsize=(7,5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribuzione dei residui")
plt.xlabel("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "residuals_distribution.png"))
plt.close()

print(f"Analisi multivariata completata. Grafici salvati in: {img_dir}")
