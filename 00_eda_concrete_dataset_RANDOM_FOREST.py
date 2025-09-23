import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Carica il dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path, sep=";")

# Rimuovo la colonna ridondante 'Unnamed: 0'
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Rimuovo righe con valori mancanti
df = df.dropna()

# Variabili predittive e target
X = df.drop(columns=["Strength", "id"])
y = df["Strength"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predizioni
y_pred = rf.predict(X_test)

# Metriche di valutazione
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("R²:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# Directory salvataggio immagini
img_dir = "ml_random_forest"
os.makedirs(img_dir, exist_ok=True)

# Feature importance
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "feature_importance.png"))
plt.close()

# Predicted vs Actual
plt.figure(figsize=(7,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("Predicted vs Actual - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "predicted_vs_actual.png"))
plt.close()
