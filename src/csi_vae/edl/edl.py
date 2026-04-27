import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import optuna
import optuna.terminator
import torch
from edl_losses.edl import EDLLoss, edl_inference
from edl_losses.gen import GENLoss, gen_inference
from rich.logging import RichHandler

from csi_vae.edl.edl_trainer import EDLTrainer
from csi_vae.edl.eval import eval_edl_model
from csi_vae.edl.gen_trainer import GENTrainer
from csi_vae.jobs import JobSettings, dataset, fusion, vae
from csi_vae.jobs.job import make_dataloader
from csi_vae.launcher import make_study
from csi_vae.studies import get_best_model, read_studies

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False, rich_tracebacks=True)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Disable Optuna warnings
optuna.logging.disable_default_handler()
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

N_TRIALS = 100
LAUNCH_DIR = Path("out/flat_conv_full")
RESULTS_DIR = Path("out/edl")
WEIGHTS_DIR = Path("weights/fusion")
OOD_DATASET = Path("dataset/S1b.h5")
OBJECTIVE_ALPHA = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = JobSettings()

# Best model
studies = read_studies(LAUNCH_DIR)
best_model = get_best_model(studies)
gaussians = [
    vae.SingleAntenna(
        settings.window_size,
        settings.n_subcarriers,
        best_model.n_gaussians,
        vae.CONV_SPECS[best_model.params["conv_layers_spec"]],
    ).to(DEVICE)
    for _ in range(settings.n_antennas)
]
delayed_fusion = fusion.Delayed(
    gaussians,
    best_model.n_gaussians,
    settings.n_activities,
    best_model.params["n_fusion_layers"],
    best_model.params["fusion_dropout"],
).to(DEVICE)

model_path = (
    WEIGHTS_DIR
    / f"l{best_model.n_gaussians}"
    / f"t{best_model.trial_number}"
    / f"s{best_model.seed}"
    / "delayed_fusion.pt"
)
delayed_fusion.load_state_dict(torch.load(model_path))

# Dataset
batch_size = 2 ** best_model.params["batch_size_exp"]

train_ds, val_ds, test_ds = dataset.load(
    dataset_path=Path(settings.dataset_path),
    window_size=settings.window_size,
    n_activities=settings.n_activities,
    stride=settings.stride,
)
full_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])

train_dl = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, seed=best_model.seed)
val_dl = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)
test_dl = make_dataloader(test_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)
full_dl = make_dataloader(full_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)

ood_train_ds, ood_val_ds, ood_test_ds = dataset.load(
    dataset_path=OOD_DATASET,
    window_size=settings.window_size,
    n_activities=settings.n_activities,
    stride=settings.stride,
)
ood_ds = torch.utils.data.ConcatDataset([ood_train_ds, ood_val_ds, ood_test_ds])
ood_dl = make_dataloader(ood_ds, batch_size=batch_size, shuffle=False, seed=best_model.seed)


def _edl_objective(trial: optuna.Trial) -> float:
    edl_fusion = fusion.Delayed(
        gaussians,
        best_model.n_gaussians,
        settings.n_activities,
        best_model.params["n_fusion_layers"],
        best_model.params["fusion_dropout"],
    )
    loss_type = trial.suggest_categorical("loss_type", ["sse", "ce", "mse"])
    should_anneal = trial.suggest_categorical("should_anneal", [True, False])
    if should_anneal:
        anneal_epochs = trial.suggest_int("annealing_epochs", 1, settings.n_epochs // 2)
        edl_loss = EDLLoss(loss_type=loss_type, anneal_epochs=anneal_epochs)  # pyright: ignore[reportArgumentType]
    else:
        beta = trial.suggest_float("beta", 0.0, 1.0)
        edl_loss = EDLLoss(loss_type=loss_type, beta=beta)  # pyright: ignore[reportArgumentType]

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
        device=DEVICE,
    )
    edl_trainer.train(settings.n_epochs)

    accuracy, unc_correct, unc_wrong, _, _ = eval_edl_model(
        model=edl_fusion,
        inference_fn=edl_inference,
        dataloader=test_dl,
        device=DEVICE,
    )

    return OBJECTIVE_ALPHA * accuracy + (1 - OBJECTIVE_ALPHA) * (unc_correct - unc_wrong)


def _gen_objective(trial: optuna.Trial) -> float:
    gen_fusion = fusion.Delayed(
        gaussians,
        best_model.n_gaussians,
        settings.n_activities,
        best_model.params["n_fusion_layers"],
        best_model.params["fusion_dropout"],
    ).to(DEVICE)

    should_anneal = trial.suggest_categorical("should_anneal", [True, False])
    if should_anneal:
        anneal_epochs = trial.suggest_int("annealing_epochs", 1, settings.n_epochs // 2)
        gen_loss = GENLoss(anneal_epochs=anneal_epochs)
    else:
        beta = trial.suggest_float("beta", 0.0, 1.0)
        gen_loss = GENLoss(beta=beta)

    gen_trainer = GENTrainer(
        gen_fusion,
        train_dl,
        val_dl,
        fusion.TrainerParams(
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            early_stop_patience=settings.early_stop_patience,
            early_stop_warmup_epochs=settings.early_stop_warmup_epochs,
        ),
        criterion=gen_loss,
        device=DEVICE,
    )
    gen_trainer.train(settings.n_epochs)

    accuracy, unc_correct, unc_wrong, _, _ = eval_edl_model(
        model=gen_fusion,
        inference_fn=gen_inference,
        dataloader=test_dl,
        device=DEVICE,
    )

    return OBJECTIVE_ALPHA * accuracy + (1 - OBJECTIVE_ALPHA) * (unc_correct - unc_wrong)


def _run_study(study_name: str, objective_fn: Callable) -> None:
    study = make_study(study_name, str(RESULTS_DIR), settings.seed)

    # Check how many trials are already complete to avoid re-running them if the launcher is restarted
    already_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = N_TRIALS - already_done

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
