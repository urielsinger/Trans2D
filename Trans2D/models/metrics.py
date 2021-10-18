from typing import Optional, Any, Callable

import torch
from torch import Tensor
from torchmetrics.retrieval import RetrievalMetric


class RetrievalHIT(RetrievalMetric):
    """
    Hit@k metric for retrieval tasks
    """
    def __init__(
        self,
        empty_target_action: str = 'neg',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        k: int = None
    ):
        super().__init__(
            empty_target_action=empty_target_action,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")
        self.k = k

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        indices = torch.argsort(preds, descending=True)[:self.k]
        return 1 * (target[indices].sum() > 0)