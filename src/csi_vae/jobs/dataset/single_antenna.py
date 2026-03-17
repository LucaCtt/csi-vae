import torch
from torch.utils.data import Dataset

from csi_vae.jobs.dataset.multi_antenna import MultiAntenna


class SingleAntenna(Dataset):
    """Wraps MultiAntenna to expose a single antenna's data, avoiding redundant data loads."""

    def __init__(self, dataset: MultiAntenna, antenna_select: int) -> None:
        """Initialize the SingleAntenna dataset.

        Arguments:
            dataset: The source MultiAntenna dataset.
            antenna_select: Index of the antenna to select.

        Raises:
            ValueError: If antenna_select is out of range.

        """
        n_antennas = dataset[0][0].shape[0]
        if not 0 <= antenna_select < n_antennas:
            msg = f"antenna_select={antenna_select} out of range for {n_antennas} antennas"
            raise ValueError(msg)

        self.__dataset = dataset
        self.__antenna_select = antenna_select

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.__dataset[idx]
        return x[self.__antenna_select], y
