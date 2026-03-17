import numpy as np
import torch
from torch.utils.data import Dataset


class MultiAntenna(Dataset):
    """CSI dataset with lazy sliding-window indexing over multiple antennas."""

    def __init__(
        self,
        csi_mats: list[np.ndarray],
        window_size: int,
        stride: int,
    ) -> None:
        """Initialize the MultiAntenna dataset.

        Arguments:
            csi_mats: List of CSI matrices, one per activity, each with shape
                (n_antennas, n_samples, n_subcarriers).
            window_size: Number of time steps per window.
            stride: Step size between consecutive windows.

        Raises:
            ValueError: If any matrix has fewer time steps than window_size,
                or if csi_mats is empty.

        """
        if not csi_mats:
            msg = "csi_mats must not be empty"
            raise ValueError(msg)

        if window_size < 1 or stride < 1:
            msg = f"window_size and stride must be >= 1, got {window_size=}, {stride=}"
            raise ValueError(msg)

        self.__window_size = window_size

        self.__data: list[np.ndarray] = []
        self.__index_map: list[tuple[int, int]] = []

        # Load files once, build index map
        for label, csi in enumerate(csi_mats):
            n_samples = csi.shape[1]
            if n_samples < window_size:
                msg = f"Activity {label}: n_samples={n_samples} < window_size={window_size}"
                raise ValueError(msg)

            self.__data.append(csi)

            # Build lazy sliding-window index
            for offset in range(0, n_samples - window_size + 1, stride):
                self.__index_map.append((label, offset))

    def __len__(self) -> int:
        return len(self.__index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        label, start = self.__index_map[idx]
        csi = self.__data[label]

        window = csi[:, start : start + self.__window_size, :]

        x = torch.from_numpy(window)
        return x, label
