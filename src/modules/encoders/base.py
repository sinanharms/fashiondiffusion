from abc import abstractmethod
from typing import Any, List, TypeVar

import torch
from torch import nn

Torch = TypeVar("torch.Tensor")


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x: Torch, **kwargs) -> List[Torch]:
        raise NotImplementedError

    def decode(self, z: Torch, **kwargs) -> Any:
        raise NotImplementedError

    def sample(self, z, batch_size: int, device: str, **kwargs) -> Torch:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *x: Torch) -> Torch:
        pass

    @abstractmethod
    def loss(self, *x: Any, **kwargs) -> Torch:
        pass
