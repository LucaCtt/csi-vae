from pathlib import Path

import optuna


def read_studies(launch_dir: Path) -> list[optuna.Study]:
    """Read all Optuna studies from the specified launch directory and return them as a list of DataFrames.

    Arguments:
        launch_dir (Path): The directory where the Optuna study SQLite files are located.

    Returns:
        list[optuna.Study]: A list of Optuna Study objects loaded from the SQLite files in the launch directory.

    """
    studies_files = sorted([f.name for f in launch_dir.iterdir() if f.is_file() and f.suffix == ".sqlite"])
    return [
        optuna.load_study(study_name=study.split(".")[0], storage=f"sqlite:///{launch_dir / study}")
        for study in studies_files
    ]


def get_best_model(studies: list[optuna.Study]) -> dict:
    """Return the best model across all studies based on the highest seed accuracy.

    Arguments:
        studies: A list of Optuna Study objects, each containing trial data for different hyperparameter configurations.

    Returns:
        A dictionary containing the details of the best model.

    """
    best_models_per_study: list[dict] = []

    for i, study in enumerate(studies):
        study_df = study.trials_dataframe()
        completed = study_df[study_df["state"] == "COMPLETE"].copy()
        study_best = {
            "n_gaussians": i + 1,
            "trial_number": 0,
            "trial_value": 0.0,
            "seed": 0,
            "best_seed_accuracy": 0.0,
            "params": {},
        }

        for _, trial in completed.iterrows():
            accuracies_per_seed = trial["user_attrs_accuracies"]

            best_seed = max(accuracies_per_seed, key=accuracies_per_seed.get)
            best_accuracy = float(accuracies_per_seed[str(best_seed)])

            if trial["value"] > study_best["trial_value"]:
                study_best = {
                    "n_gaussians": i + 1,
                    "trial_number": trial["number"],
                    "trial_value": trial["value"],
                    "seed": int(best_seed),
                    "best_seed_accuracy": best_accuracy,
                    "accuracies_per_seed": accuracies_per_seed,
                    "params": trial.filter(like="params_").rename(lambda x: x.replace("params_", "")).to_dict(),
                }

        best_models_per_study.append(study_best)

    return max(best_models_per_study, key=lambda x: x["best_seed_accuracy"])
