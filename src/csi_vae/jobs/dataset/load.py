from pathlib import Path
from string import ascii_uppercase
from typing import cast

import h5py
import numpy as np

from csi_vae.jobs.dataset.multi_antenna import MultiAntenna

_SPLITS = ("train", "val", "test")


def load(
    dataset_path: Path,
    window_size: int,
    n_activities: int,
    stride: int,
) -> tuple[MultiAntenna, MultiAntenna, MultiAntenna]:
    """Load the CSI train/val/test datasets from an HDF5 file.

    Arguments:
        dataset_path: Path to the HDF5 dataset file.
        window_size: Window size for CSI samples.
        n_activities: Number of activities to load (max 26, keyed A--Z).
        stride: Stride of the sliding window.

    Returns:
        A tuple of (train, val, test) MultiAntenna datasets.

    Raises:
        ValueError: If n_activities is out of range.
        KeyError: If expected groups or activity keys are missing from the file.

    """
    activity_keys = list(ascii_uppercase[:n_activities])
    split_mats: dict[str, list[np.ndarray]] = {split: [] for split in _SPLITS}

    with h5py.File(dataset_path, "r") as f:
        missing_splits = [s for s in _SPLITS if s not in f]
        if missing_splits:
            msg = f"HDF5 file is missing expected groups: {missing_splits}"
            raise KeyError(msg)

        for split in _SPLITS:
            group = cast("h5py.Group", f[split])
            missing_keys = [k for k in activity_keys if k not in group]
            if missing_keys:
                msg = f"Split '{split}' is missing activity keys: {missing_keys}"
                raise KeyError(msg)

            for key in activity_keys:
                split_mats[split].append(np.array(group[key]))

    return tuple(  # type: ignore[return-value]
        MultiAntenna(csi_mats=split_mats[split], window_size=window_size, stride=stride) for split in _SPLITS
    )
