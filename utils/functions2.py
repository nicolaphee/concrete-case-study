def 
    scoring = {
        "rmse": make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred, )),
        "mae": make_scorer(mean_absolute_error),
        "r2": make_scorer(r2_score)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    results = []
    all_scores = []  # per distribuzioni metriche

    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        if sample_weighting:
            # calcola pesi sul target train
            sample_weights = compute_sample_weights(y_train)
            fit_params = make_fit_params(pipe, sample_weights)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True, params=fit_params)
        else:
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        # Calcola composite score (per bilanciare accuratezza e overfitting)
        train_rmse_mean = scores["train_rmse"].mean()
        train_rmse_std = scores["train_rmse"].std()
        test_rmse_mean = scores["test_rmse"].mean()
        test_rmse_std = scores["test_rmse"].std()
        overfit_gap = test_rmse_mean - train_rmse_mean
        composite_score = test_rmse_mean + alpha * overfit_gap + beta * test_rmse_std

        # Salva medie e deviazioni standard
        results.append({
            "Model": name,
            "MAE mean": scores["test_mae"].mean(),
            "MAE std": scores["test_mae"].std(),
            "R2 mean": scores["test_r2"].mean(),
            "R2 std": scores["test_r2"].std(),
            "RMSE mean": test_rmse_mean,
            "RMSE std": test_rmse_std,
            "RMSE mean train": train_rmse_mean,
            "RMSE std train": train_rmse_std,
            "Overfit gap": overfit_gap,
            "Composite score": composite_score,
        })
        
        # Salva tutte le fold per grafici
        for i in range(cv.get_n_splits()):
            all_scores.append({
                "Model": name, "Fold": i+1,
                "RMSE": scores["train_rmse"][i],
                "MAE": scores["train_mae"][i],
                "R2": scores["train_r2"][i],
                "Set": "Training"
            })
            all_scores.append({
                "Model": name, "Fold": i+1,
                "RMSE": scores["test_rmse"][i],
                "MAE": scores["test_mae"][i],
                "R2": scores["test_r2"][i],
                "Set": "Validation"
            })


    # Tabella riassuntiva
    # results_df = pd.DataFrame(results).sort_values("RMSE mean")
    results_df = pd.DataFrame(results).sort_values("Composite score")
    results_df.to_csv(os.path.join(img_dir, "models_summary.csv"), index=False)
    print(results_df)