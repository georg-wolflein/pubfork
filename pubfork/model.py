import re
from collections import ChainMap
from typing import Any, Dict, Mapping, Optional, Tuple, Sequence, Type, Union
from functools import partial

import re
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from omegaconf import DictConfig

from .metrics import create_metrics_for_target
from .utils import sanitize
from .relative import DistanceAwareMultiheadAttention


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
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
        self.add_zero_attn = add_zero_attn

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
        if self.q_proj.bias is not None:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.v_proj.bias is not None:
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
        q = self.q_proj(query)  # [Batch, SeqLen, Dims]
        k = self.k_proj(key)  # [Batch, SeqLen, Dims]
        v = self.v_proj(value)  # [Batch, SeqLen, Dims]

        if self.add_zero_attn:
            q = torch.cat([q, torch.zeros(batch_size, 1, self.kdim).type_as(q)], dim=1)
            k = torch.cat([k, torch.zeros(batch_size, 1, self.kdim).type_as(k)], dim=1)

        q = q.reshape(
            *q.shape[:2], self.num_heads, self.kdim // self.num_heads
        )  # [Batch, SeqLen, Head, Dims]
        k = k.reshape(
            *k.shape[:2], self.num_heads, self.kdim // self.num_heads
        )  # [Batch, SeqLen, Head, Dims]
        v = v.reshape(
            *v.shape[:2], self.num_heads, self.vdim // self.num_heads
        )  # [Batch, SeqLen, Head, Dims]

        q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k = k.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        v = v.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        # Scaled dot product attention
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / d_k**0.5
        attention = F.softmax(attn_logits, dim=-1)  # [Batch, Head, SeqLen, SeqLen]
        if self.add_zero_attn:
            # Remove zeroed out tokens
            attention = attention[:, :, :-1, :-1]

        # Apply dropout
        dropout_attention = self.dropout(attention)

        # Apply attention to values
        values = torch.matmul(dropout_attention, v)

        # Unify heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)

        if need_weights:
            return values, attention
        return (values,)


MHA = Union[MultiheadAttention, DistanceAwareMultiheadAttention, nn.MultiheadAttention]


class MHAWrapper(nn.Module):

    def __init__(self, mha: MHA):
        super().__init__()
        self.mha = mha
        self.add_tile_position_kwarg = (
            "tile_positions" in mha.forward.__code__.co_varnames
        )

    def forward(self, *args, tile_positions=None, **kwargs):
        if self.add_tile_position_kwarg:
            return self.mha(*args, tile_positions=tile_positions, **kwargs)[0]
        else:
            return self.mha(*args, tile_positions=tile_positions, **kwargs)[0]


class MilTransformer(nn.Module):
    def __init__(
        self,
        d_features: int,
        targets: Sequence[DictConfig],
        agg: str = "max",  # "mean" or "max"
        num_layers: int = 1,
        num_heads: int = 4,
        do_linear_proj: bool = False,
        do_initial_linear_proj: bool = False,
        hidden_dim=128,
        att_dropout=0.1,
        linear_dropout=0.1,
        add_zero_attn=False,
        layer_norm=True,
        mha1: Type[MHA] = MultiheadAttention,
        mhas: Type[MHA] = MultiheadAttention,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.agg = agg
        self.linear_dropout = linear_dropout
        self.layer_norm = layer_norm

        # self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(
                hidden_dim if do_initial_linear_proj else d_features
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
            )
        else:
            self.layer_norms = [None for _ in range(num_layers - 1)]

        if do_initial_linear_proj:
            self.linear1 = nn.Linear(d_features, hidden_dim)
            self.msa1 = MHAWrapper(
                mha1(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=att_dropout,
                    batch_first=True,
                    kdim=hidden_dim,
                    vdim=hidden_dim,
                    add_zero_attn=add_zero_attn,
                )
            )
        else:
            self.linear1 = None
            self.msa1 = MHAWrapper(
                mhas(
                    embed_dim=d_features,
                    num_heads=num_heads,
                    dropout=att_dropout,
                    batch_first=True,
                    kdim=hidden_dim,
                    vdim=hidden_dim,
                    add_zero_attn=add_zero_attn,
                )
            )
        self.msas = nn.ModuleList(
            [
                MHAWrapper(
                    mhas(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=att_dropout,
                        batch_first=True,
                        kdim=hidden_dim,
                        vdim=hidden_dim,
                        add_zero_attn=add_zero_attn,
                    )
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.linears = (
            nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
            )
            if do_linear_proj
            else [None for _ in range(num_layers - 1)]
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
        if self.linear1:
            x = self.linear1(x)
            if self.linear_dropout:
                x = F.dropout(x, p=self.linear_dropout)
            x = F.relu(x)
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.msa1(x, x, x, tile_positions=tile_positions)
        for layer_norm, linear, msa in zip(self.layer_norms, self.linears, self.msas):
            if linear:
                x = linear(x)
                if self.linear_dropout:
                    x = F.dropout(x, p=self.linear_dropout)
                x = F.relu(x)
            # Linear bug: apply the following instead of layer_norm
            # if linear:
            #     x = linear(x)
            if layer_norm:
                x = layer_norm(x)
            x = msa(x, x, x, tile_positions=tile_positions)

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
                nn.ModuleDict(
                    {
                        sanitize(target.column): create_metrics_for_target(target)
                        for target in self.targets
                    }
                ),
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

                target_metrics.update(
                    logits[target.column],
                    targets[target.column].argmax(dim=1),
                )
                self.log_dict(
                    {
                        f"{step_name}_{target.column}_{name}": metric
                        for name, metric in target_metrics.items()
                    },
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
