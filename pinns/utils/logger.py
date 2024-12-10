from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from pinns.utils.io_utils import get_root

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print('Wandb module not available. Skipping wandb initialization.')


class Logger:
    def __init__(self, savepath: Path, config: Dict[str, Any], use_wandb: bool = False):
        self.config = config
        self.savepath = get_root() / savepath
        self.savepath.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(**config) if use_wandb and WANDB_AVAILABLE else None
        self.loss_history = defaultdict(list)

    def save_loss(self, loss_values: Dict[str, float]):
        loss_values = {key: np.log10(value) for key, value in loss_values.items()}
        for key, value in loss_values.items():
            self.loss_history[key].append(value)

        if self.run is not None:
            self.run.log(loss_values)

    def save_history(self):
        np.save(self.savepath / 'loss_history.npy', [self.loss_history])

    def save_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        epoch: Optional[int] = None,
    ):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if epoch is not None:
            path = self.savepath / f'epoch_{epoch}'
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = self.savepath
        torch.save(checkpoint, path / filename)
        if self.run is not None:
            self.run.log_artifact(path)

    @staticmethod
    def load_checkpoint(
        path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
    ) -> int:
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
