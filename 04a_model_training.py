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

# Definiamo i modelli richiesti
models = {
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000),
    "SVR": SVR(kernel="rbf", C=100, gamma="scale"),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
}

results = {}

# Addestramento e valutazione modelli
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    # Usiamo StandardScaler per modelli sensibili alla scala
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
