from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Carica il dataset
file_path = "dataset_imputed.csv"
df = pd.read_csv(file_path, sep=";")

# Separiamo features e target
X = df.drop(columns=["Strength"])
y = df["Strength"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rifacciamo senza XGBoost per evitare problemi di import e pesantezza
models = {
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000),
    "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=1, n_jobs=-1),
    "MLP": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training and evaluating {name}...")
    if name in ["Ridge", "Lasso", "ElasticNet", "SVR", "KNN", "MLP"]:
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        pipeline = Pipeline([("model", model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

results_df = pd.DataFrame(results).T
results_df


import matplotlib.pyplot as plt
import os

img_dir = "model_comparison_plots"
os.makedirs(img_dir, exist_ok=True)

# Barplot R2
plt.figure(figsize=(8,5))
results_df["R2"].sort_values().plot(kind="barh", color="skyblue")
plt.title("Confronto R² tra modelli")
plt.xlabel("R²")
plt.ylabel("Modelli")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()
plt.tight_layout()
out_path = os.path.join(img_dir, f"R2_comparison.png")
plt.savefig(out_path)
plt.close()

# Barplot RMSE
plt.figure(figsize=(8,5))
results_df["RMSE"].sort_values().plot(kind="barh", color="salmon")
plt.title("Confronto RMSE tra modelli")
plt.xlabel("RMSE")
plt.ylabel("Modelli")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()
plt.tight_layout()

out_path = os.path.join(img_dir, f"RMSE_comparison.png")
plt.savefig(out_path)
plt.close()

# Scatter plot valori reali vs predetti per il migliore modello (Random Forest)
best_model_name = results_df["R2"].idxmax()
best_model = models[best_model_name]

# Se serve scaler lo aggiungiamo
if best_model_name in ["Ridge", "Lasso", "ElasticNet", "SVR", "KNN", "MLP"]:
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", best_model)])
else:
    pipeline = Pipeline([("model", best_model)])

pipeline.fit(X_train, y_train)
y_pred_best = pipeline.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.5, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valori reali")
plt.ylabel("Predizioni")
plt.title(f"Confronto valori reali vs predetti\nModello migliore: {best_model_name}")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
plt.tight_layout()

out_path = os.path.join(img_dir, f"best_true_vs_pred.png")
plt.savefig(out_path)
plt.close()
