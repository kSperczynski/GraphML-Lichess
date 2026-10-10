# Reproducing and comparing the MLP pilot

- Run the notebook from top to bottom in the repository Python environment.
- config.json freezes data identity, feature order, architecture, split seed, training seeds, and evaluation rules before test evaluation.
- player_splits.csv maps every player to train, validation, or test. Join by username; do not rely on CSV row numbers.
- predictions_by_player.csv contains separate predictions for each model, split, and MLP seed. Do not treat three seed predictions as three independent players.
- metrics_by_run.csv contains individual-run metrics. metrics_summary.csv contains model-level summaries.
- preprocessing.joblib stores training-fitted preprocessing and the feature order; ridge_model.joblib stores Ridge; mlp_seed_*.pt store MLP state dictionaries.
- environment.json records runtime package versions. training_runs.csv records selected epochs and validation metrics.
- The notebook has already evaluated the frozen pilot on the held-out test set. Do not tune this pilot using its test results.
- This split is proposed for team reuse; agreement with teammates is still required. Graph construction and temporal leakage require separate checks.
- Checkpoints reflect current aggregate-feature data and must be regenerated when the task or feature definitions change.
