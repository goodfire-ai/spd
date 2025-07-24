import torch
import torchvision
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

InputTensor = Float[Tensor, "1 28 28"]
LabelTensor = Int[Tensor, ""]


class NoiseDataset(Dataset[tuple[InputTensor, LabelTensor]]):
    """Uniform noise images"""

    def __init__(self, n: int, seed: int, device: str = "cpu") -> None:
        rng: torch.Generator = torch.Generator().manual_seed(seed)
        self.data: Float[Tensor, "n 1 28 28"] = torch.rand((n, 1, 28, 28), generator=rng).to(device)
        self.fake_label: Float[Tensor, " n"] = torch.zeros(n, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[InputTensor, LabelTensor]:
        return self.data[idx], self.fake_label[idx]

    def dataloader(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> DataLoader[tuple[InputTensor, LabelTensor]]:
        """Create a DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class GpuMNIST(torchvision.datasets.MNIST):
    """MNIST cached on device"""

    def __init__(self, train: bool, device: str = "cpu") -> None:
        super().__init__(root="data", train=train, download=True)
        self.data: Float[Tensor, "n 1 28 28"] = self.data.unsqueeze(1).float().div(255.0).to(device)
        self.targets: Int[Tensor, " n"] = self.targets.to(device)

    def __getitem__(self, idx: int) -> tuple[InputTensor, LabelTensor]:
        return self.data[idx], self.targets[idx]

    def dataloader(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> DataLoader[tuple[InputTensor, LabelTensor]]:
        """Create a DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
