import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import optuna
import optuna.terminator
import torch
from edl_losses.edl import EDLLoss
from edl_losses.gen import GENLoss, gen_inference
from rich.logging import RichHandler

from csi_vae.edl.edl_settings import EDLSettings
from csi_vae.edl.edl_trainer import EDLTrainer
from csi_vae.edl.eval import eval_edl_model
from csi_vae.edl.gen_trainer import GENTrainer, GENTrainerParams
from csi_vae.jobs import dataset, fusion, vae
from csi_vae.jobs.job import init_rng, make_dataloader
from csi_vae.studies import get_best_model, make_study, read_studies

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False, rich_tracebacks=True)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Disable Optuna warnings
optuna.logging.disable_default_handler()
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

settings = EDLSettings()

# Best model
studies = read_studies(settings.launch_studies_dir)
best_model = get_best_model(studies)

# Set RNG seed for reproducibility
init_rng(best_model.seed)

# Load best model weights
gaussians = [
    vae.SingleAntenna(
        settings.window_size,
        settings.n_subcarriers,
        best_model.n_gaussians,
        vae.CONV_SPECS[best_model.params["conv_layers_spec"]],
    )
    for _ in range(settings.n_antennas)
]
delayed_fusion = fusion.Delayed(
    gaussians,
    best_model.n_gaussians,
    settings.n_activities,
    best_model.params["n_fusion_layers"],
    best_model.params["fusion_dropout"],
)
model_path = (
    Path(settings.launch_weights_dir)
    / f"l{best_model.n_gaussians}"
    / f"t{best_model.trial_number}"
    / f"s{best_model.seed}"
    / "delayed_fusion.pt"
)
delayed_fusion.load_state_dict(torch.load(model_path))

# Datasets
train_ds, val_ds, test_ds = dataset.load(
    dataset_path=Path(settings.dataset_path),
    window_size=settings.window_size,
    n_activities=settings.n_activities,
    stride=settings.stride,
)
batch_size = 2 ** best_model.params["batch_size_exp"]
train_dl = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, seed=best_model.seed)
val_dl = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)
test_dl = make_dataloader(test_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)


def _edl_objective(trial: optuna.Trial) -> float:
    edl_fusion = fusion.Delayed(
        gaussians,
        best_model.n_gaussians,
        settings.n_activities,
        best_model.params["n_fusion_layers"],
        best_model.params["fusion_dropout"],
    )
    anneal_epochs = trial.suggest_int("anneal_epochs", 1, 2 * settings.n_epochs // 3)
    edl_loss = EDLLoss(anneal_epochs=anneal_epochs)
    edl_trainer = EDLTrainer(
        edl_fusion,
        train_dl,
        val_dl,
        fusion.TrainerParams(
            lr=best_model.params["lr"],
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        ),
        criterion=edl_loss,
    )
    edl_trainer.train(settings.n_epochs)

    out_dir = Path(settings.edl_weights_dir) / "edl" / f"t{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        edl_fusion.state_dict(),
        out_dir / "delayed_fusion.pt",
    )

    accuracy, unc_correct, unc_wrong = eval_edl_model(
        model=edl_fusion,
        inference_fn=gen_inference,  # Use gen inference which is the same as edl but uses exp instead of relu
        dataloader=test_dl,
    )
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("unc_correct", unc_correct)
    trial.set_user_attr("unc_wrong", unc_wrong)

    if accuracy < settings.objective_accuracy_threshold or unc_wrong < unc_correct:
        raise optuna.TrialPruned

    logger.info(
        "[EDL] Trial %d - Accuracy: %.4f, Unc Correct: %.4f, Unc Wrong: %.4f",
        trial.number,
        accuracy,
        unc_correct,
        unc_wrong,
    )

    return settings.objective_alpha * accuracy + (1 - settings.objective_alpha) * (unc_wrong - unc_correct)


def _gen_objective(trial: optuna.Trial) -> float:
    gen_fusion = fusion.Delayed(
        gaussians,
        best_model.n_gaussians,
        settings.n_activities,
        best_model.params["n_fusion_layers"],
        best_model.params["fusion_dropout"],
    )
    beta_mode = trial.suggest_categorical("beta_mode", ["auto", "anneal"])
    if beta_mode == "auto":
        gen_loss = GENLoss()
    else:
        anneal_epochs = trial.suggest_int("anneal_epochs", 1, 2 * settings.n_epochs // 3)
        gen_loss = GENLoss(anneal_epochs=anneal_epochs)
    gen_trainer = GENTrainer(
        gen_fusion,
        train_dl,
        val_dl,
        GENTrainerParams(
            lr=best_model.params["lr"],
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
            noise_scale=trial.suggest_float("noise_scale", 0.1, 0.5),
        ),
        criterion=gen_loss,
    )
    gen_trainer.train(settings.n_epochs)

    out_dir = Path(settings.edl_weights_dir) / "gen" / f"t{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        gen_fusion.state_dict(),
        out_dir / "delayed_fusion.pt",
    )

    accuracy, unc_correct, unc_wrong = eval_edl_model(
        model=gen_fusion,
        inference_fn=gen_inference,
        dataloader=test_dl,
    )
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("unc_correct", unc_correct)
    trial.set_user_attr("unc_wrong", unc_wrong)

    if accuracy < settings.objective_accuracy_threshold or unc_wrong - unc_correct < 0:
        raise optuna.TrialPruned

    logger.info(
        "[GEN] Trial %d - Accuracy: %.4f, Unc Correct: %.4f, Unc Wrong: %.4f",
        trial.number,
        accuracy,
        unc_correct,
        unc_wrong,
    )

    return settings.objective_alpha * accuracy + (1 - settings.objective_alpha) * (unc_wrong - unc_correct)


def _run_study(study_name: str, objective_fn: Callable) -> None:
    study = make_study(study_name, settings.edl_studies_dir, best_model.seed)

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


if __name__ == "__main__":
    _run_study("edl", _edl_objective)
    _run_study("gen", _gen_objective)
