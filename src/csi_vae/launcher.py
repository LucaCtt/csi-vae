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

from csi_vae.aws import JobSubmitter, MessagesQueue
from csi_vae.jobs import JobSettings, MessageType
from csi_vae.launcher_settings import LauncherSettings

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


def _make_study(study_name: str, storage_dir: str | None, seed: int) -> optuna.Study:
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


def _get_params(trial: optuna.Trial, settings: LauncherSettings) -> dict[str, str | int | float]:
    """Extract the parameters for a trial as a dictionary."""
    lr = trial.suggest_float("lr", settings.lr.min, settings.lr.max, log=True)
    kl_max = trial.suggest_float("kl_max", settings.kl_max.min, settings.kl_max.max, log=True)
    n_fusion_layers = trial.suggest_int("n_fusion_layers", settings.n_fusion_layers.min, settings.n_fusion_layers.max)
    conv_layers_spec = trial.suggest_categorical("conv_layers_spec", settings.conv_layers_spec.values)

    bs_exp_min = int(math.log2(settings.batch_size.min))
    bs_exp_max = int(math.log2(settings.batch_size.max))
    batch_size = 2 ** trial.suggest_int("batch_size_exp", bs_exp_min, bs_exp_max)

    cc_mult_min = settings.conv_channels.min // 8
    cc_mult_max = settings.conv_channels.max // 8
    conv_channels = 8 * trial.suggest_int("conv_channels_mult", cc_mult_min, cc_mult_max)

    return {
        "lr": lr,
        "kl_max": kl_max,
        "batch_size": batch_size,
        "conv_channels": conv_channels,
        "conv_layers_spec": conv_layers_spec,
        "n_fusion_layers": n_fusion_layers,
    }


class Launcher:
    """Launcher for running Optuna studies on AWS Batch."""

    def __init__(self, settings: LauncherSettings) -> None:
        """Initialize the launcher with the given settings.

        Arguments:
            settings: The LauncherSettings object containing configuration for the launcher.

        """
        self.__settings = settings
        self.__submitter = JobSubmitter(settings.batch_job_queue, settings.batch_job_definition, settings.region_name)
        self.__queue = MessagesQueue(settings.region_name)
        self.__seeds = _generate_seeds(settings.starter_seed, settings.n_seeds_per_trial)

    def launch(self) -> None:
        """Launch the Optuna studies, iterating over latent dimensions and running trials."""
        self.__queue.create(self.__settings.launch_name)

        best_accuracy: float = float("-inf")
        patience_counter = 0

        for latent_dim in range(self.__settings.latent_dim.min, self.__settings.latent_dim.max + 1):
            accuracy = self.__run_study(latent_dim)

            delta = accuracy - best_accuracy
            if delta < self.__settings.min_accuracy_delta:
                logger.info("[L=%d] Accuracy did not improve sufficiently.", latent_dim)

                patience_counter += 1
                if patience_counter >= self.__settings.latent_dim_patience:
                    logger.info("[L=%d] Patience exhausted. Stopping search.", latent_dim)
                    break
            else:
                logger.info("[L=%d] Accuracy improved by %.4f.", latent_dim, delta)
                patience_counter = 0
                best_accuracy = accuracy

    def __run_study(self, latent_dim: int) -> float:
        """Run all trials for a given latent_dim. Returns the best accuracy achieved.

        Arguments:
            latent_dim: The latent dimension to run the study for.

        Returns:
            The best median accuracy achieved across all trials for this latent dimension.

        """
        study_name = f"l{latent_dim}"
        study = _make_study(study_name, self.__settings.storage_dir, self.__settings.starter_seed)

        # Check how many trials are already complete to avoid re-running them if the launcher is restarted
        already_done = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = self.__settings.n_trials - already_done
        if remaining <= 0:
            logger.info("[L=%d] Study already complete, skipping.", latent_dim)
            return study.best_value

        logger.info("[L=%d] Starting study '%s' (%d trials remaining).", latent_dim, study_name, remaining)

        emmr_evaluator = optuna.terminator.EMMREvaluator()
        median_error_evaluator = optuna.terminator.MedianErrorEvaluator(emmr_evaluator)
        terminator = optuna.terminator.Terminator(emmr_evaluator, median_error_evaluator)
        callbacks = [optuna.terminator.TerminatorCallback(terminator)]
        study.optimize(
            lambda trial: self.__run_trial(trial, latent_dim),
            n_trials=remaining,
            callbacks=callbacks,
            catch=(optuna.exceptions.TrialPruned, TimeoutError, RuntimeError),
        )

        # After optimization, log the best trial and return its value
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            logger.warning("[L=%d] No completed trials.", latent_dim)
            return 0.0

        logger.info(
            "[L=%d] Best trial is #%d with median_accuracy=%.4f, params=%s.",
            latent_dim,
            study.best_trial.number,
            study.best_trial.value,
            study.best_trial.params,
        )
        return study.best_trial.value if study.best_trial.value is not None else 0.0

    def __run_trial(self, trial: optuna.Trial, latent_dim: int) -> float:
        """Run a single Optuna trial."""
        params = _get_params(trial, self.__settings)

        jobs = []
        for seed in self.__seeds:
            trial_settings = JobSettings(
                region_name=self.__settings.region_name,
                bucket_name=self.__settings.bucket_name,
                bucket_key=f"{self.__settings.launch_name}/l{latent_dim}/t{trial.number}/s{seed}",
                queue_url=self.__queue.url,
                trial_number=trial.number,
                latent_dim=latent_dim,
                seed=seed,
                **params,  # pyright: ignore[reportArgumentType]
            )

            job_name = f"{self.__settings.launch_name}_l{latent_dim}_t{trial.number}_s{seed}"
            job_id = self.__submitter.submit(job_name, trial_settings.model_dump())
            jobs.append(job_id)
            logger.debug("[L=%d][T=%d][S=%d] Submitted job %s.", latent_dim, trial.number, seed, job_id)

        logger.info(
            "[L=%d][T=%d] Submitted %d jobs with params=%s.",
            latent_dim,
            trial.number,
            len(self.__seeds),
            trial.params,
        )

        try:
            results = self.__poll_results(latent_dim, trial.number)
        except Exception:
            for job_id in jobs:
                self.__submitter.terminate(job_id)
            raise

        median_accuracy = statistics.median(list(results.values()))
        quantiles = statistics.quantiles(list(results.values()))
        trial.set_user_attr("accuracies", results)
        trial.set_user_attr("accuracy_p25", float(quantiles[0]))
        trial.set_user_attr("accuracy_p75", float(quantiles[2]))

        logger.info(
            "[L=%d][T=%d] Trial finished with median_accuracy=%.4f.",
            latent_dim,
            trial.number,
            median_accuracy,
        )

        return median_accuracy

    def __poll_results(
        self,
        latent_dim: int,
        trial_number: int,
    ) -> dict[int, float]:
        """Poll the messages queue for results from the given trial until all seeds have reported or too many collapses.

        Arguments:
            latent_dim: The latent dimension for this trial (used to filter messages for this study).
            trial_number: The Optuna trial number (used to filter messages for this trial).

        Returns:
            A list of accuracies reported by the seeds for this trial.

        """
        results: dict[int, float] = {}
        collapses = 0
        start = time.monotonic()

        while len(results) + collapses < len(self.__seeds):
            time.sleep(self.__settings.poll_interval)

            if time.monotonic() - start > self.__settings.poll_timeout:
                msg = "Timed out waiting for seed results."
                raise TimeoutError(msg)

            messages = self.__queue.pop(max_messages=len(self.__seeds))
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
                    results[seed] = message["accuracy"]

                elif message_type == MessageType.COLLAPSE:
                    collapses += 1
                    logger.warning(
                        "[L=%d][T=%d][S=%d] Job collapsed (%d total).",
                        latent_dim,
                        trial_number,
                        seed,
                        collapses,
                    )
                    if collapses > self.__settings.max_pruned_seeds:
                        logger.warning(
                            "[L=%d][T=%d][S=%d] Too many collapses, trial pruned.",
                            latent_dim,
                            trial_number,
                            seed,
                        )

                        msg = f"More than {self.__settings.max_pruned_seeds} seeds collapsed (got {collapses})."
                        raise optuna.TrialPruned(msg)

                elif message_type == MessageType.ERROR:
                    msg = f"Failed with error: {message.get('error', 'Unknown error')}"
                    raise RuntimeError(msg)

        return results

    def cleanup(self) -> None:
        """Clean up resources used by the launcher."""
        self.__queue.destroy()


def run_launcher(settings: LauncherSettings | None = None) -> None:
    """Run the launcher with the given settings (or defaults if None)."""
    settings = LauncherSettings() if settings is None else settings
    launcher = Launcher(settings)

    try:
        launcher.launch()
    finally:
        launcher.cleanup()


if __name__ == "__main__":
    run_launcher()
