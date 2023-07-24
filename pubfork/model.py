# %%
import re
from collections import ChainMap
from typing import Any, Dict, Mapping, Optional, Tuple, Sequence
from functools import partial

import re
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn
from torchmetrics.utilities.data import dim_zero_cat
from omegaconf import DictConfig


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Unsupported arguments
        assert batch_first is True
        assert add_bias_kv is False
        assert add_zero_attn is False
        assert device is None
        assert dtype is None

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        assert (
            kdim % num_heads == 0 and vdim % num_heads == 0
        ), "kdims and vdims must be divisible by num_heads"
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(embed_dim, kdim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kdim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, vdim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        need_weights: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = query.shape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(
            batch_size, seq_length, self.num_heads, self.kdim // self.num_heads
        )
        k = k.reshape(
            batch_size, seq_length, self.num_heads, self.kdim // self.num_heads
        )
        v = v.reshape(
            batch_size, seq_length, self.num_heads, self.vdim // self.num_heads
        )

        q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k = k.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        v = v.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        # Scaled dot product attention
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / d_k**0.5
        attention = F.softmax(attn_logits, dim=-1)

        dropout_attention = self.dropout(attention)
        values = torch.matmul(dropout_attention, v)

        # Unify heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)

        if need_weights:
            return values, attention
        return (values,)


class MilTransformer(nn.Module):
    def __init__(
        self,
        d_features: int,
        targets: Sequence[DictConfig],
        agg: str = "max",  # "mean" or "max"
    ) -> None:
        super().__init__()
        self.targets = targets
        self.agg = agg

        # self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        hidden_dim = 64

        self.msa1 = MultiheadAttention(
            embed_dim=d_features,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.msa2 = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )

        self.heads = nn.ModuleDict(
            {
                sanitize(target.column): nn.Linear(
                    in_features=hidden_dim,
                    out_features=len(target.classes)
                    if target.type == "categorical"
                    else 1,
                )
                for target in targets
            }
        )

    def forward(
        self,
        tile_tokens: torch.Tensor,
        tile_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = tile_tokens.shape

        # shape: [bs, seq_len, d_model]
        # tile_tokens = self.projector(tile_tokens)

        x = tile_tokens
        x = self.msa1(x, x, x)[0]
        x = self.msa2(x, x, x)[0]

        # Aggregate the tile tokens
        if self.agg == "mean":
            slide_tokens = x.mean(dim=-2)
        elif self.agg == "max":
            slide_tokens = x.max(dim=-2).values
        else:
            raise NotImplementedError(f"Unknown aggregation method {self.agg}")

        # Apply the corresponding head to each slide-level token
        logits = {
            target.column: self.heads[sanitize(target.column)](slide_tokens)
            for target in self.targets
        }

        return logits


def create_metrics_for_target(target) -> torchmetrics.MetricCollection:
    if target.type == "categorical":
        return torchmetrics.MetricCollection(
            {
                f"auroc": SafeMulticlassAUROC(num_classes=len(target.classes)),
                f"aurocs": SafeMulticlassAUROC(
                    num_classes=len(target.classes), average="macro"
                ),
            }
        )
    else:
        raise NotImplementedError(f"Unknown target type {target.type}")


class LitMilClassificationMixin(pl.LightningModule):
    """Makes a module into a multilabel, multiclass Lightning one"""

    def __init__(
        self,
        *,
        targets: list,
        # Other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # So we don't get unused parameter warnings

        self.learning_rate = learning_rate
        self.targets = targets

        for step_name in ["train", "val", "test"]:
            setattr(
                self,
                f"{step_name}_target_metrics",
                {
                    sanitize(target.column): create_metrics_for_target(target)
                    for target in self.targets
                },
            )

        self.save_hyperparameters()

    def step(self, batch: Tuple[Tensor, Tensor], step_name=None):
        feats, coords, targets = batch
        logits = self(feats, coords)

        # Calculate the cross entropy loss for each target, then sum them
        loss = sum(
            F.cross_entropy(
                (l := logits[target.column]),
                targets[target.column].type_as(l),
                weight=torch.tensor(target.weights).type_as(l)
                if target.weights
                else None,
            )
            for target in self.targets
        )

        if step_name:
            self.log(
                f"{step_name}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            # Update target-wise metrics
            for target in self.targets:
                target_metrics = getattr(self, f"{step_name}_target_metrics")[
                    sanitize(target.column)
                ]

                is_na = (targets[target.column] == 0).all(dim=1)
                target_metrics.update(
                    logits[target.column][~is_na],
                    targets[target.column][~is_na].argmax(dim=1),
                )
                for name, metric in target_metrics.items():
                    self.log(
                        f"{step_name}_{target.column}_{name}",
                        metric,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 2:
            feats, positions = batch
        else:
            feats, positions, _ = batch
        logits = self(feats, positions)

        softmaxed = {
            target_label: torch.softmax(x, 1) for target_label, x in logits.items()
        }
        return softmaxed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def sanitize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


class SafeMulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """A Multiclass AUROC that doesn't blow up when no targets are given"""

    def compute(self) -> torch.Tensor:
        # Add faux entry if there are none so far
        if len(self.preds) == 0:
            self.update(torch.zeros(1, self.num_classes), torch.zeros(1).long())
        elif len(dim_zero_cat(self.preds)) == 0:
            self.update(
                torch.zeros(1, self.num_classes).type_as(self.preds[0]),
                torch.zeros(1).long().type_as(self.target[0]),
            )
        return super().compute()
