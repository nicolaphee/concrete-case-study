random_state = 42

n_iter = 20  # numero di iterazioni per RandomizedSearchCV
apply_feature_eng = True # se True, applica feature engineering
log_transform_target = False  # se True, applica log-transform al target
sample_weighting = False  # se True, usa pesi campione
use_simple_imputer = False # se True, imputa tutto con la mediana
use_clipping = False # se True, applica clipping agli outlier
mlflow_logging = True # se True, abilita il logging su MLflow
