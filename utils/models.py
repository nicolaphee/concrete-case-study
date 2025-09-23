import sys
sys.path.append("./utils")  # per importare funzioni da ../utils
from params import random_state

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# advanced boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.neural_network import MLPRegressor

from scipy.stats import randint, uniform

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


models = {
    # # Baseline
    # "MeanPredictor": DummyRegressor(strategy="mean"),

    # Baseline lineari
    # "Ridge": Ridge(alpha=1.0, random_state=random_state),
    "Ridge": Ridge(alpha=10.0, random_state=random_state),
    # "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
    "Lasso": Lasso(alpha=0.05, random_state=random_state, max_iter=10000),
    # "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state, max_iter=10000),
    "BayesianRidge": BayesianRidge(),

    # Non lineari classici
    # "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7))]),
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance"))]),
    # "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1))]),
    # "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=1, epsilon=0.2))]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=0.1, epsilon=0.3))]),

    # Tree-based
    # "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state),
    # "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5, random_state=random_state),
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5, random_state=random_state),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, random_state=random_state),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=7, min_samples_split=7, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, learning_rate=0.05, random_state=random_state),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=7, learning_rate=0.05, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=7, min_samples_split=7, learning_rate=0.05, random_state=random_state),

    # Boosting avanzati
    # "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=random_state),
    # "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.025, max_depth=5, random_state=random_state),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=random_state),
    # "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, verbose=0, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.025, depth=5, early_stopping_rounds=100, verbose=0, random_state=random_state),
    # "CatBoost": CatBoostRegressor(n_estimators=1000, learning_rate=0.01, depth=4, early_stopping_rounds=100, verbose=0, random_state=random_state),
    "CatBoost": CatBoostRegressor(n_estimators=500, learning_rate=0.03, depth=3, early_stopping_rounds=100, l2_leaf_reg=3, verbose=0, random_state=random_state),

    # Rete neurale
    # "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))])
    # "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, alpha=0.001, early_stopping=True, validation_fraction=0.2, random_state=random_state))]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, alpha=0.01, early_stopping=True, validation_fraction=0.2, random_state=random_state))]),

    # PLS: riduce dimensionalità → minore rischio di overfit con molte variabili collineari
    "PLSRegression": PLSRegression(n_components=5),  # 5 componenti è un buon default

    # Regressori robusti: resistenti a outlier e rumore
    "Huber": Pipeline([("scaler", RobustScaler()),("huber", HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=10000))]),
    "RANSAC": Pipeline([("scaler", RobustScaler()),("ransac", RANSACRegressor(min_samples=0.5, max_trials=100, random_state=random_state))]),
    "TheilSen": Pipeline([("scaler", RobustScaler()),("theilsen", TheilSenRegressor(max_subpopulation=10000, random_state=random_state))]),

    # # Gaussian Process con kernel RBF regolarizzato
    # "GaussianProcess": Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("gpr", GaussianProcessRegressor(
    #         kernel=C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0),
    #         alpha=1.0,          # regolarizzazione (rumore) relativamente alta
    #         normalize_y=True,
    #         random_state=random_state
    #     ))
    # ]),

    # # Bagging con regressori lineari regolarizzati
    # "BaggingRidge": BaggingRegressor(
    #     estimator=Pipeline([("scaler", StandardScaler()),("ridge", Ridge(alpha=1.0, random_state=random_state))]),
    #     n_estimators=50,
    #     max_samples=0.8,
    #     max_features=0.8,
    #     random_state=random_state
    # ),
    # "BaggingLasso": BaggingRegressor(
    #     estimator=Pipeline([("scaler", StandardScaler()),("lasso", Lasso(alpha=0.01, random_state=random_state, max_iter=10000))]),
    #     n_estimators=50,
    #     max_samples=0.8,
    #     max_features=0.8,
    #     random_state=random_state
    # ),

}


param_grids = {
    "Ridge": {"model__alpha": uniform(1e-4, 10)},
    "Lasso": {"model__alpha": uniform(1e-4, 1)},
    "ElasticNet": {
        "model__alpha": uniform(1e-4, 1),
        "model__l1_ratio": uniform(0, 1),
    },
    "kNN": {
        "model__knn__n_neighbors": randint(5, 30),
        "model__knn__weights": ["uniform", "distance"],
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
        "model__n_estimators": randint(300, 800),      # numero alberi
        "model__max_depth": [5, 10, 15],         # None = senza limite
        "model__max_features": ["sqrt", 0.5, 0.7],     # meno feature = più robusto
        "model__min_samples_leaf": randint(2, 10),     # ↑ min_leaf = meno overfit
        "model__min_samples_split": randint(2, 10),    # ↑ split threshold
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
    "LightGBM": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [-1, 4, 6],
        "model__num_leaves": randint(15, 64),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "CatBoost": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__depth": randint(3, 6),
        "model__l2_leaf_reg": uniform(1, 10),
    },
    "MLP": {
        "model__mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "model__mlp__alpha": uniform(1e-4, 1e-2),
        "model__mlp__learning_rate_init": uniform(1e-4, 1e-2),
        "model__mlp__early_stopping": [True],
    },
    "PLSRegression": {
        "model__n_components": randint(2, 10),
    },
    "Huber": {
        "model__huber__alpha": uniform(1e-4, 1.0),
        "model__huber__epsilon": uniform(1.2, 0.8),  # 1.2–2.0
    },
    "RANSAC": {
        "model__ransac__min_samples": uniform(0.4, 0.5),  # 0.4–0.9
        "model__ransac__max_trials": randint(50, 200),
        "model__ransac__residual_threshold": uniform(1.0, 5.0),
    },
    "TheilSen": {
        "model__theilsen__max_subpopulation": [1000, 5000, 10000],
        "model__theilsen__tol": uniform(1e-4, 1e-2),
    },
    # "GaussianProcess": {
    #     "model__gpr__alpha": uniform(1e-2, 10.0),
    #     "model__gpr__kernel": [
    #         C(1.0, (1e-2, 1e2)) * RBF(length_scale=l)
    #         for l in [0.5, 1.0, 2.0, 5.0]
    #     ],
    # },
    # "BaggingRidge": {
    #     "model__estimator__ridge__alpha": uniform(1e-3, 10.0),
    #     "model__n_estimators": randint(20, 100),
    #     "model__max_samples": uniform(0.5, 0.5),   # 0.5–1.0
    #     "model__max_features": uniform(0.5, 0.5),
    # },
    # "BaggingLasso": {
    #     "model__estimator__lasso__alpha": uniform(1e-3, 1.0),
    #     "model__n_estimators": randint(20, 100),
    #     "model__max_samples": uniform(0.5, 0.5),
    #     "model__max_features": uniform(0.5, 0.5),
    # },
}