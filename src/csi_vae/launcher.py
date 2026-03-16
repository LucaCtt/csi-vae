import logging
import math
import random
import statistics
import time
import warnings
from pathlib import Path

import optuna
import optuna.terminator
from optuna.trial import TrialState
from rich.logging import RichHandler

from csi_vae.aws import MessagesQueue, TrialSubmitter
from csi_vae.launcher_settings import LauncherSettings
from csi_vae.trial import MessageType, TrialSettings

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False, rich_tracebacks=True)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Disable Optuna warnings
optuna.logging.disable_default_handler()
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def _generate_seeds(starter_seed: int, n_seeds: int) -> list[int]:
    """Generate a fixed list of seeds from a starter seed.

    Arguments:
        starter_seed: The seed to use for the random number generator.
        n_seeds: The number of seeds to generate.

    Returns:
        A list of n_seeds integers to use as seeds for the trials.

    """
    rng = random.Random(starter_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_seeds)]


def _make_study(study_name: str, journal_path: str | None) -> optuna.Study:
    """Create (or load) an Optuna study backed by a journal file.

    Arguments:
        study_name: The name of the study to create or load.
        journal_path: The path to the journal file to use for storage.
            If None, the study will be created without persistent storage.

    Returns:
        An Optuna Study object.

    """
    if journal_path:
        Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
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
        direction="maximize",
        load_if_exists=True,
    )


def _poll_results(
    queue: MessagesQueue,
    latent_dim: int,
    seeds: list[int],
    trial_number: int,
    max_pruned_seeds: int,
    poll_timeout: float,
    poll_interval: float,
) -> list[float]:
    """Poll the messages queue for results from the given trial until all seeds have reported or too many collapses.

    Arguments:
        queue: The MessagesQueue to poll for results.
        latent_dim: The latent dimension for this trial (used to filter messages for this study).
        seeds: The list of seeds that were run for this trial (used to track which results are still pending).
        trial_number: The Optuna trial number (used to filter messages for this trial).
        max_pruned_seeds: The maximum number of seed collapses allowed before pruning the trial.
        poll_timeout: The maximum number of seconds to wait for results before giving up.
        poll_interval: The number of seconds to wait between polling attempts.

    Returns:
        A list of accuracy values for each successful seed.

    """
    results: list[float] = []
    collapses = 0
    start = time.monotonic()

    while len(results) + collapses < len(seeds):
        time.sleep(poll_interval)

        if time.monotonic() - start > poll_timeout:
            msg = "Timed out waiting for seed results."
            raise TimeoutError(msg)

        messages = queue.pop(max_messages=len(seeds))

        for message in messages:
            if message["trial_number"] != trial_number or message["latent_dim"] != latent_dim:
                continue

            start = time.monotonic()  # reset timeout timer upon receiving a relevant message
            seed = message["seed"]
            message_type = message["type"]

            if message_type == MessageType.STARTING:
                logger.debug("[L=%d][T=%d][S=%d] Job started.", latent_dim, trial_number, seed)

            elif message_type == MessageType.SUCCESS:
                logger.info(
                    "[L=%d][T=%d][S=%d] Job succeeded with accuracy=%.4f.",
                    latent_dim,
                    trial_number,
                    seed,
                    message["accuracy"],
                )
                results.append(message["accuracy"])

            elif message_type == MessageType.COLLAPSE:
                collapses += 1
                logger.warning(
                    "[L=%d][T=%d][S=%d] Job collapsed (%d total).",
                    latent_dim,
                    trial_number,
                    seed,
                    collapses,
                )
                if collapses > max_pruned_seeds:
                    logger.warning(
                        "[L=%d][T=%d][S=%d] Too many collapses, trial pruned.",
                        latent_dim,
                        trial_number,
                        seed,
                    )

                    msg = f"More than {max_pruned_seeds} seeds collapsed (got {collapses})."
                    raise optuna.TrialPruned(msg)

            elif message_type == MessageType.ERROR:
                msg = f"Seed {seed} failed with error: {message.get('error', 'Unknown error')}"
                raise RuntimeError(msg)

    return results


def _run_trial(
    trial: optuna.Trial,
    latent_dim: int,
    seeds: list[int],
    settings: LauncherSettings,
    submitter: TrialSubmitter,
    queue: MessagesQueue,
) -> float:
    """Run a single Optuna trial."""
    # lr: log-uniform float
    lr = trial.suggest_float("lr", settings.lr.min, settings.lr.max, log=True)

    # kl_max: uniform float
    kl_max = trial.suggest_float("kl_max", settings.kl_max.min, settings.kl_max.max, log=True)

    # batch_size: powers of 2 — sample exponent, then map back
    bs_exp_min = int(math.log2(settings.batch_size.min))
    bs_exp_max = int(math.log2(settings.batch_size.max))
    batch_size = 2 ** trial.suggest_int("batch_size_exp", bs_exp_min, bs_exp_max)

    # conv_channels: multiples of 8 — sample multiplier, then map back
    cc_mult_min = settings.conv_channels.min // 8
    cc_mult_max = settings.conv_channels.max // 8
    conv_channels = 8 * trial.suggest_int("conv_channels_mult", cc_mult_min, cc_mult_max)

    conv_layers_spec = trial.suggest_categorical("conv_layers_spec", settings.conv_layers_spec.values)

    n_fusion_layers = trial.suggest_int("n_fusion_layers", settings.n_fusion_layers.min, settings.n_fusion_layers.max)

    jobs = []
    # Submit one job per seed
    for seed in seeds:
        trial_settings = TrialSettings(
            study_name=settings.launch_name,
            trial_number=trial.number,
            queue_url=queue.url,
            region_name=settings.region_name,
            bucket_name=settings.bucket_name,
            latent_dim=latent_dim,
            seed=seed,
            lr=lr,
            kl_max=kl_max,
            batch_size=batch_size,
            conv_channels=conv_channels,
            conv_layers_spec=conv_layers_spec,
            n_fusion_layers=n_fusion_layers,
        )
        job_id = submitter.submit(trial_settings)
        jobs.append(job_id)
        logger.debug("[L=%d][T=%d][S=%d] Submitted job %s.", latent_dim, trial.number, seed, job_id)

    logger.info("[L=%d][T=%d] Submitted %d jobs with params=%s.", latent_dim, trial.number, len(seeds), trial.params)

    try:
        results = _poll_results(
            queue,
            latent_dim,
            seeds,
            trial.number,
            settings.max_pruned_seeds,
            settings.poll_timeout,
            settings.poll_interval,
        )
    except Exception:
        for job_id in jobs:
            submitter.terminate(job_id)
        raise

    median_accuracy = statistics.median(results)
    quantiles = statistics.quantiles(results, n=4)
    trial.set_user_attr("accuracies", results)
    trial.set_user_attr("accuracy_p25", float(quantiles[0]))
    trial.set_user_attr("accuracy_p75", float(quantiles[2]))

    optuna.terminator.report_cross_validation_scores(trial, results)

    logger.info(
        "[L=%d][T=%d] Trial finished with median accuracy=%.4f.",
        latent_dim,
        trial.number,
        median_accuracy,
    )

    return median_accuracy


def _run_study(
    latent_dim: int,
    seeds: list[int],
    settings: LauncherSettings,
    submitter: TrialSubmitter,
    queue: MessagesQueue,
) -> float:
    """Run all trials for a given latent_dim. Returns the best accuracy achieved."""
    study_name = f"{settings.launch_name}_l{latent_dim}"
    study = _make_study(study_name, f"{settings.journal_dir}/l{latent_dim}.sqlite")

    already_done = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    remaining = settings.n_trials - already_done
    if remaining <= 0:
        logger.info("[L=%d] Study already complete, skipping.", latent_dim)
        return study.best_value

    logger.info("[L=%d] Starting study '%s' (%d trials remaining).", latent_dim, study_name, remaining)

    def objective(trial: optuna.Trial) -> float:
        return _run_trial(trial, latent_dim, seeds, settings, submitter, queue)

    terminator = optuna.terminator.Terminator(optuna.terminator.EMMREvaluator())
    callbacks = [optuna.terminator.TerminatorCallback(terminator)]
    study.optimize(objective, n_trials=remaining, callbacks=callbacks)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        logger.warning("[L=%d] No completed trials.", latent_dim)
        return 0.0

    best = study.best_trial
    logger.info(
        "[L=%d] Best trial is #%d with medianaccuracy=%.4f, params=%s.",
        latent_dim,
        best.number,
        best.value,
        best.params,
    )
    return best.value if best.value is not None else 0.0


def run_launcher(settings: LauncherSettings | None = None) -> None:
    """Run the launcher with the given settings (or defaults if None)."""
    settings = LauncherSettings() if settings is None else settings

    submitter = TrialSubmitter(settings.batch_job_queue, settings.batch_job_definition, settings.region_name)
    queue = MessagesQueue(settings.region_name)
    queue.create(settings.launch_name)

    seeds = _generate_seeds(settings.starter_seed, settings.n_seeds_per_trial)
    logger.info("Generated %d seeds from starter seed %d.", len(seeds), settings.starter_seed)

    try:
        previous_best_accuracy: float | None = None
        patience_counter = 0

        for latent_dim in range(settings.latent_dim.min, settings.latent_dim.max + 1):
            best_accuracy = _run_study(latent_dim, seeds, settings, submitter, queue)

            if previous_best_accuracy is not None:
                delta = best_accuracy - previous_best_accuracy
                logger.info(
                    "[L=%d] Accuracy delta vs previous: %+.4f (threshold: %f)",
                    latent_dim,
                    delta,
                    settings.min_accuracy_delta,
                )

                if delta < settings.min_accuracy_delta:
                    patience_counter += 1
                    logger.info(
                        "[L=%d] Accuracy did not improve sufficiently. Patience counter: %d.",
                        latent_dim,
                        patience_counter,
                    )
                    if patience_counter >= settings.latent_dim_patience:
                        logger.info("[L=%d] Patience exhausted. Stopping search.", latent_dim)
                        break
                else:
                    patience_counter = 0

            previous_best_accuracy = best_accuracy
    finally:
        queue.destroy()


if __name__ == "__main__":
    run_launcher()
