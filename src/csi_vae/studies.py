from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna


@dataclass
class StudyResult:
    """Data class to store the results of a single Optuna studyl."""

    n_gaussians: int
    trial_number: int
    trial_value: float
    seed: int
    best_seed_accuracy: float
    accuracies_per_seed: dict[str, float]
    params: dict[str, Any]


def make_study(study_name: str, storage_dir: str | None, seed: int) -> optuna.Study:
    """Create (or load) an Optuna study backed by a journal file.

    Arguments:
        study_name: The name of the study to create or load.
        storage_dir: The directory to use for storage. If None, the study will be created without persistent storage.
        seed: The seed to use for the random number generator.

    Returns:
        An Optuna Study object.

    """
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        journal_path = f"{storage_dir}/{study_name}.sqlite"
    else:
        journal_path = ":memory:"

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{journal_path}",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed),
        direction="maximize",
        load_if_exists=True,
    )


def read_studies(launch_dir: str) -> list[optuna.Study]:
    """Read all Optuna studies from the specified launch directory and return them as a list of DataFrames.

    Arguments:
        launch_dir (Path): The directory where the Optuna study SQLite files are located.

    Returns:
        list[optuna.Study]: A list of Optuna Study objects loaded from the SQLite files in the launch directory.

    """
    studies_files = sorted([f.name for f in Path(launch_dir).iterdir() if f.is_file() and f.suffix == ".sqlite"])
    return [
        optuna.load_study(study_name=study.split(".")[0], storage=f"sqlite:///{Path(launch_dir) / study}")
        for study in studies_files
    ]


def get_best_model(studies: list[optuna.Study]) -> StudyResult:
    """Return the best model across all studies based on the highest seed accuracy.

    Arguments:
        studies: A list of Optuna Study objects, each containing trial data for different hyperparameter configurations.

    Returns:
        StudyResult: A dataclass containing the details of the best model found across all studies.

    """
    best_models_per_study: list[StudyResult] = []

    for i, study in enumerate(studies):
        study_df = study.trials_dataframe()
        completed = study_df[study_df["state"] == "COMPLETE"].copy()
        study_best = StudyResult(
            n_gaussians=i + 1,
            trial_number=0,
            trial_value=0.0,
            seed=0,
            best_seed_accuracy=0.0,
            accuracies_per_seed={},
            params={},
        )

        for _, trial in completed.iterrows():
            accuracies_per_seed = trial["user_attrs_accuracies"]

            best_seed = max(accuracies_per_seed, key=accuracies_per_seed.get)
            best_accuracy = float(accuracies_per_seed[str(best_seed)])

            if trial["value"] > study_best.trial_value:
                study_best = StudyResult(
                    n_gaussians=i + 1,
                    trial_number=trial["number"],
                    trial_value=trial["value"],
                    seed=int(best_seed),
                    best_seed_accuracy=best_accuracy,
                    accuracies_per_seed=accuracies_per_seed,
                    params=trial.filter(like="params_").rename(lambda x: x.replace("params_", "")).to_dict(),
                )

        best_models_per_study.append(study_best)

    return max(best_models_per_study, key=lambda x: x.best_seed_accuracy)
