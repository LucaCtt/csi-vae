import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path

import optuna
import optuna.terminator
import torch
from edl_losses.edl import edl_inference
from edl_losses.gen import gen_inference
from rich.logging import RichHandler

from csi_vae.edl.edl_settings import EDLSettings
from csi_vae.edl.edl_trainer import EDLTrainer, EDLTrainerParams
from csi_vae.edl.evaluator import EDLEvaluator
from csi_vae.edl.gen_trainer import GENTrainer, GENTrainerParams
from csi_vae.jobs import dataset, fusion, vae
from csi_vae.jobs.job import init_rng, make_dataloader
from csi_vae.studies import get_best_model, make_study, read_studies

ALPHA = 0.5
"""Weight for balancing accuracy and uncertainty in the objective function."""

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False, rich_tracebacks=True)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Disable Optuna warnings
optuna.logging.disable_default_handler()
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

settings = EDLSettings()

# Load best model Optuna results
studies = read_studies(settings.launch_studies_dir)
wi_har_results = get_best_model(studies)

# Set RNG seed for reproducibility
init_rng(wi_har_results.seed)

# Load best model weights
wi_har_gaussians = [
    vae.SingleAntenna(
        settings.window_size,
        settings.n_subcarriers,
        wi_har_results.n_gaussians,
        vae.CONV_SPECS[wi_har_results.params["conv_layers_spec"]],
    )
    for _ in range(settings.n_antennas)
]
wi_har_model = fusion.Delayed(
    wi_har_gaussians,
    wi_har_results.n_gaussians,
    settings.n_activities,
    wi_har_results.params["n_fusion_layers"],
    wi_har_results.params["fusion_dropout"],
)
wi_har_model_path = (
    Path(settings.launch_weights_dir)
    / f"l{wi_har_results.n_gaussians}"
    / f"t{wi_har_results.trial_number}"
    / f"s{wi_har_results.seed}"
    / "delayed_fusion.pt"
)
wi_har_model.load_state_dict(torch.load(wi_har_model_path))

# Datasets
train_ds, val_ds, test_ds = dataset.load(
    dataset_path=Path(settings.dataset_path),
    window_size=settings.window_size,
    n_activities=settings.n_activities,
    stride=settings.stride,
)
batch_size = 2 ** wi_har_results.params["batch_size_exp"]
train_dl = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, seed=wi_har_results.seed)
val_dl = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, seed=wi_har_results.seed)
test_dl = make_dataloader(test_ds, batch_size=batch_size, shuffle=False, seed=wi_har_results.seed)


def _edl_objective(trial: optuna.Trial) -> float:
    """Objective function for EDL study.

    Trains an EDL model with hyperparameters from the trial and evaluates it on the test set.

    Arguments:
        trial (optuna.Trial): The Optuna trial object containing hyperparameters.

    Returns:
        float: The computed objective value combining accuracy and uncertainty metrics.

    """
    edl_fusion = fusion.Delayed(
        wi_har_gaussians,
        wi_har_results.n_gaussians,
        settings.n_activities,
        wi_har_results.params["n_fusion_layers"],
        wi_har_results.params["fusion_dropout"],
    )
    anneal_epochs = trial.suggest_int("anneal_epochs", 10, 3 * settings.n_epochs // 4)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    edl_trainer = EDLTrainer(
        edl_fusion,
        train_dl,
        val_dl,
        EDLTrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            loss_type="sse",
            beta="anneal",
            anneal_epochs=anneal_epochs,
        ),
    )
    edl_trainer.train(settings.n_epochs)

    out_dir = Path(settings.edl_weights_dir) / "edl" / f"t{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        edl_fusion.state_dict(),
        out_dir / "delayed_fusion.pt",
    )

    accuracy, unc_correct, unc_wrong, cohens_d = EDLEvaluator(
        model=edl_fusion,
        inference_fn=edl_inference,
        dataloader=test_dl,
    ).evaluate()
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("unc_correct", unc_correct)
    trial.set_user_attr("unc_wrong", unc_wrong)
    trial.set_user_attr("cohens_d", cohens_d)

    if cohens_d < 0:
        raise optuna.TrialPruned

    logger.info(
        "[EDL] Trial %d - Accuracy: %.4f, Unc Correct: %.4f, Unc Wrong: %.4f, Cohen's d: %.4f",
        trial.number,
        accuracy,
        unc_correct,
        unc_wrong,
        cohens_d,
    )
    cohens_d_normalized = float(math.tanh(cohens_d / 2))

    return ALPHA * accuracy + (1 - ALPHA) * cohens_d_normalized


def _gen_objective(trial: optuna.Trial) -> float:
    """Objective function for GEN study.

    Trains a GEN model with hyperparameters from the trial and evaluates it on the test set.

    Arguments:
        trial (optuna.Trial): The Optuna trial object containing hyperparameters.

    Returns:
        float: The computed objective value combining accuracy and uncertainty metrics.

    """
    gen_fusion = fusion.Delayed(
        wi_har_gaussians,
        wi_har_results.n_gaussians,
        settings.n_activities,
        wi_har_results.params["n_fusion_layers"],
        wi_har_results.params["fusion_dropout"],
    )

    beta = trial.suggest_categorical("beta_mode", ["auto", "anneal"])
    anneal_epochs = trial.suggest_int("anneal_epochs", 10, 3 * settings.n_epochs // 4) if beta == "anneal" else 10
    gan_hidden_dim = trial.suggest_int("gan_hidden_dim", 32, 256, step=32)
    gan_lr = trial.suggest_float("gan_lr", 1e-5, 1e-3, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    gen_trainer = GENTrainer(
        gen_fusion,
        train_dl,
        val_dl,
        GENTrainerParams(
            lr=lr,
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            beta=beta,  # pyright: ignore[reportArgumentType]
            anneal_epochs=anneal_epochs,
            gan_hidden_dim=gan_hidden_dim,
            gan_lr=gan_lr,
        ),
    )
    gen_trainer.train(settings.n_epochs)

    out_dir = Path(settings.edl_weights_dir) / "gen" / f"t{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        gen_fusion.state_dict(),
        out_dir / "delayed_fusion.pt",
    )

    accuracy, unc_correct, unc_wrong, cohens_d = EDLEvaluator(
        model=gen_fusion,
        inference_fn=gen_inference,
        dataloader=test_dl,
    ).evaluate()
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("unc_correct", unc_correct)
    trial.set_user_attr("unc_wrong", unc_wrong)
    trial.set_user_attr("cohens_d", cohens_d)

    if cohens_d < 0:
        raise optuna.TrialPruned

    logger.info(
        "[GEN] Trial %d - Accuracy: %.4f, Unc Correct: %.4f, Unc Wrong: %.4f, Cohen's d: %.4f",
        trial.number,
        accuracy,
        unc_correct,
        unc_wrong,
        cohens_d,
    )

    cohens_d_normalized = float(math.tanh(cohens_d / 2))

    return ALPHA * accuracy + (1 - ALPHA) * cohens_d_normalized


def _run_study(study_name: str, objective_fn: Callable) -> None:
    """Run an Optuna study with the given objective function and study name.

    Arguments:
        study_name (str): Name of the study (e.g., "edl" or "gen").
        objective_fn (Callable): The objective function to optimize.

    """
    study = make_study(study_name, settings.edl_studies_dir, wi_har_results.seed)

    # Check how many trials are already complete to avoid re-running them if the launcher is restarted
    already_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = settings.n_trials - already_done

    logger.info("Starting %s study (%d trials remaining).", study_name, remaining)

    emmr_evaluator = optuna.terminator.EMMREvaluator()
    median_error_evaluator = optuna.terminator.MedianErrorEvaluator(emmr_evaluator)
    terminator = optuna.terminator.Terminator(emmr_evaluator, median_error_evaluator)
    callbacks = [optuna.terminator.TerminatorCallback(terminator)]
    study.optimize(
        objective_fn,
        n_trials=remaining,
        callbacks=callbacks,
        catch=(optuna.exceptions.TrialPruned, TimeoutError, RuntimeError),
    )


def run_edl() -> None:
    """Run both EDL and GEN studies."""
    _run_study("edl", _edl_objective)
    _run_study("gen", _gen_objective)


if __name__ == "__main__":
    run_edl()
