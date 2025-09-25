from utils.params_classifiers import random_state

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform



models = {
    # Baseline
    "Dummy": DummyClassifier(strategy="most_frequent"),

    # Lineari
    "LogisticRegression": LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs", random_state=random_state),

    # Non lineari classici
    "kNN": Pipeline([("scaler", StandardScaler()), 
                     ("knn", KNeighborsClassifier(n_neighbors=10, weights="distance"))]),

    "SVC": Pipeline([("scaler", StandardScaler()), 
                     ("svc", SVC(C=1, kernel="rbf", probability=True, random_state=random_state))]),

    # Tree-based
    "DecisionTree": DecisionTreeClassifier(max_depth=None, random_state=random_state),
    "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=None, random_state=random_state),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, random_state=random_state),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=random_state),

    # Boosting avanzati
    "XGBoost": XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, random_state=random_state, use_label_encoder=False, eval_metric="mlogloss"),
    # "LightGBM": LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=-1, random_state=random_state),
    # "CatBoost": CatBoostClassifier(n_estimators=500, learning_rate=0.03, depth=6, l2_leaf_reg=3, verbose=0, random_state=random_state),

    # Rete neurale
    "MLP": Pipeline([("scaler", StandardScaler()),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=random_state))]),
}

param_grids = {
    "LogisticRegression": {
        "model__C": uniform(0.01, 10),
        "model__penalty": ["l2"],
    },
    "kNN": {
        "model__knn__n_neighbors": randint(3, 20),
        "model__knn__weights": ["uniform", "distance"],
    },
    "SVC": {
        "model__svc__C": [0.1, 1, 10],
        "model__svc__kernel": ["linear", "rbf"],
        "model__svc__gamma": ["scale", 0.1, 0.01],
    },
    "DecisionTree": {
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": randint(2, 10),
        "model__min_samples_leaf": randint(1, 10),
    },
    "RandomForest": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [5, 10, 15, None],
        "model__max_features": ["sqrt", 0.5],
        "model__min_samples_leaf": randint(2, 10),
    },
    "ExtraTrees": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": [5, 10, 15, None],
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
    "LightGBM": {
        "model__n_estimators": randint(500, 1500),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__max_depth": [-1, 4, 6],
        "model__num_leaves": randint(15, 64),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
    },
    "CatBoost": {
        "model__n_estimators": randint(300, 800),
        "model__learning_rate": uniform(0.01, 0.1),
        "model__depth": [4, 6, 8],
        "model__l2_leaf_reg": uniform(1, 5),
    },
    "MLP": {
        "model__mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "model__mlp__alpha": uniform(1e-4, 1e-2),
        "model__mlp__learning_rate_init": uniform(1e-4, 1e-2),
        "model__mlp__early_stopping": [True],
    },
}
