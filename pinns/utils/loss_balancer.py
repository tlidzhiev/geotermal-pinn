from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer

import pinns
import pinns.utils


class LossBalancer:
    """
    Implementation of adaptive loss balancing for multi-objective optimization.
    Based on: https://arxiv.org/pdf/2308.08468 --- Algorithm 1
    """

    def __init__(self, loss_number: int, alpha: float = 0.9):
        self.alpha = alpha
        self.lambdas = np.ones(loss_number, dtype=np.float32)

    def update_weights(
        self,
        losses: List[torch.Tensor],
        model: nn.Module,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        assert len(losses) == len(self.lambdas)

        grad_norms = np.zeros_like(self.lambdas, dtype=np.float32)
        with torch.no_grad():
            for i, loss in enumerate(losses):
                loss_grads = pinns.utils.gradient(
                    loss,
                    model.parameters(),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_norm = torch.sqrt(sum(torch.sum(g * g) for g in loss_grads))
                grad_norms[i] = grad_norm.item()
        sum_grad_norms = np.sum(grad_norms)
        new_lambdas = sum_grad_norms / (grad_norms + eps)

        weighted_losses = [self.lambdas[i] * loss for i, loss in enumerate(losses)]
        total_loss = sum(weighted_losses)
        self.lambdas = self.alpha * self.lambdas + (1 - self.alpha) * new_lambdas
        return total_loss
