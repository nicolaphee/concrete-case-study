from utils.params import random_state

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# advanced boosting
from xgboost import XGBRegressor

# 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


models = {
    # Baseline lineari
    "Ridge": Ridge(alpha=1.0, random_state=random_state),
    "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "BayesianRidge": BayesianRidge(),

    # Non lineari classici
    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1))]),

    # Tree-based
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=random_state),

    # Boosting avanzati
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=random_state),

    # Rete neurale
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))])
}


param_grids = {
    "Ridge": {"model__alpha": uniform(1e-4, 10)},
    "Lasso": {"model__alpha": uniform(1e-4, 1)},
    "ElasticNet": {
        "model__alpha": uniform(1e-4, 1),
        "model__l1_ratio": uniform(0, 1),
    },
    "BayesianRidge": {
        "model__alpha_1": uniform(1e-5, 1e-3), 
        "model__alpha_2": uniform(1e-5, 1e-3),  
        "model__lambda_1": uniform(1e-5, 1e-3), 
        "model__lambda_2": uniform(1e-5, 1e-3), 
    },
    "SVR": {
        "model__svr__C": [0.1, 1, 10],
        "model__svr__epsilon": [0.1, 0.2, 0.3],
        "model__svr__gamma": ["scale", 0.1, 0.01],
    },
    "RandomForest": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [5, 10, 15],
        "model__max_features": ["sqrt", 0.5],
        "model__min_samples_leaf": randint(2, 10),
    },
    "ExtraTrees": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [5, 10, 15],
        "model__max_features": ["sqrt", 0.5, 0.7],
        "model__min_samples_leaf": randint(2, 10),
        "model__min_samples_split": randint(2, 10),
    },
    "GradientBoosting": {
        "model__n_estimators": randint(300, 1000),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [2, 3, 5],
        "model__subsample": uniform(0.7, 0.3),
    },
    "XGBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": randint(3, 6),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "MLP": {
        "model__mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "model__mlp__alpha": uniform(1e-4, 1e-2),
        "model__mlp__learning_rate_init": uniform(1e-4, 1e-2),
        "model__mlp__early_stopping": [True],
    },
}