from typing import List, Union

import torch


def gradient(
    outputs: torch.Tensor,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
    retain_graph: bool = False,
) -> List[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return gradients
