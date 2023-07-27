from torch import nn
import torch
from torch.nn import functional as F
from typing import Sequence

from .utils import add_zero_vector_on_dims


class DiscreteEmbeddingIndex(nn.Module):
    """Provides an embedding index for discrete values."""

    def __init__(self, num_embeddings=2):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        # x shape: num_edges x 1
        x = x * (self.num_embeddings - 1)  # assume 0 <= x <= 1
        x = torch.round(x)
        x = torch.clamp(x, 0, self.num_embeddings - 1)
        return x.long().squeeze(-1)

    @classmethod
    def index_into_embedding(cls, index, embedding: nn.Embedding):
        # return embedding(index)  # this is too slow
        return torch.index_select(embedding.weight, dim=0, index=index.view(-1)).view(
            index.shape + (-1,)
        )  # this is also too slow
        # return F.one_hot(index, embedding.num_embeddings).float() @ embedding.weight


class Embedder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        index: DiscreteEmbeddingIndex,
        trainable_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = embedding
        self.index = index
        embedding.weight.requires_grad_(trainable_embeddings)

    def forward(self, x):
        index = self.index(x)
        return self.index.index_into_embedding(index, self.embedding)


class DistanceAwareMultiheadAttention(nn.Module):
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
        # Custom arguments for relative positional encoding
        embed_keys: bool = True,
        embed_queries: bool = False,
        embed_values: bool = False,
        trainable_embeddings: bool = True,
        num_embeddings: int = 10,
    ):
        """Multihead attention with relative position representations.

        This module is a drop-in replacement for `torch.nn.MultiheadAttention` with the following differences:
        - The `embed_keys`, `embed_queries`, and `embed_values` arguments control whether the keys, queries, and values are embedded.
        - The `continuous` argument controls whether the relative positional encoding is continuous or discrete.
        - The `num_embeddings` argument controls the number of embeddings used for the relative positional encoding.
        - The `trainable_embeddings` argument controls whether the embeddings are trainable.

        The relative positional encoding is implemented as described in [1], [2] for the discrete case and [2] for the continuous case.

        [1]: https://arxiv.org/abs/1803.02155 "Self-Attention with Relative Position Representations"
        [2]: https://arxiv.org/abs/2305.10552 "Deep Multiple Instance Learning with Distance-Aware Self-Attention"
        """
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

        self.index = DiscreteEmbeddingIndex(num_embeddings=num_embeddings)

        if embed_keys:
            self.embed_k = Embedder(
                nn.Embedding(num_embeddings, kdim // num_heads),
                index=self.index,
                trainable_embeddings=trainable_embeddings,
            )
        if embed_queries:
            self.embed_q = Embedder(
                nn.Embedding(num_embeddings, kdim // num_heads),
                index=self.index,
                trainable_embeddings=trainable_embeddings,
            )
        if embed_values:
            self.embed_v = Embedder(
                nn.Embedding(num_embeddings, vdim // num_heads),
                index=self.index,
                trainable_embeddings=trainable_embeddings,
            )

        self.embed_keys = embed_keys
        self.embed_queries = embed_queries
        self.embed_values = embed_values

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

    @staticmethod
    def compute_relative_distances(
        tile_positions: torch.Tensor, max_dist: float = 100_000 * 2**0.5
    ):
        """
        Compute pairwise Euclidean distances between all pairs of positions in a tile.
        :param tile_positions: [Batch, SeqLen, 2] tensor of 2D positions
        :param max_dist: maximum distance to normalize by
        :return: [Batch, SeqLen, SeqLen] tensor of distances
        """

        # Compute pairwise differences
        diff = tile_positions.unsqueeze(2) - tile_positions.unsqueeze(1)
        # Compute pairwise distances
        dist = torch.norm(diff, dim=-1)
        print("Max dist:", dist.max())
        if max_dist:
            dist /= max_dist
        return dist

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tile_positions: torch.Tensor,
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
        # attn_logits: [Batch, Head, SeqLen, SeqLen]

        # Compute additional distance-aware terms for keys/queries
        rel_dists = self.compute_relative_distances(
            tile_positions
        )  # [Batch, SeqLen, SeqLen]

        # Term 1
        if self.embed_keys:
            rk = self.embed_k(rel_dists)  # [Batch, SeqLen, SeqLen, Dims]
            if self.add_zero_attn:
                rk = add_zero_vector_on_dims(rk, dims=(1, 2))
            rk = rk.unsqueeze(-4)  # [Batch, 1, SeqLen, SeqLen, Dims]
            q_repeat = q.unsqueeze(-2)  # [Batch, Head, SeqLen, 1, Dims]
            # A = A + (q_repeat * rk).sum(axis=-1)  # NxN
            attn_logits = attn_logits + torch.einsum("bhqrd,bhqrd->bhqr", q_repeat, rk)

        # Term 2
        if self.embed_queries:
            rq = self.embed_q(rel_dists)  # [Batch, SeqLen, SeqLen, Dims]
            if self.add_zero_attn:
                rq = add_zero_vector_on_dims(rq, dims=(1, 2))
            rq = rq.unsqueeze(-4)  # [Batch, 1, SeqLen, SeqLen, Dims]
            k_repeat = k.unsqueeze(-3)  # [Batch, Head, 1, SeqLen, Dims]
            # A = A + (k_repeat * rq).sum(axis=-1)  # NxN
            attn_logits = attn_logits + torch.einsum("bhqrd,bhqrd->bhqr", k_repeat, rq)

        # Term 3
        if self.embed_keys and self.embed_queries:
            # A = A + (q_repeat * k_repeat).sum(axis=-1)  # NxN
            attn_logits = attn_logits + torch.einsum(
                "bhqrd,bhqrd->bhqr", q_repeat, k_repeat
            )

        # Scale by sqrt(d_k)
        attn_logits = attn_logits / d_k**0.5
        attention = F.softmax(attn_logits, dim=-1)  # [Batch, Head, SeqLen, SeqLen]
        if self.add_zero_attn:
            # Remove zeroed out tokens
            attention = attention[:, :, :-1, :-1]

        # Apply dropout
        dropout_attention = self.dropout(attention)

        # Apply attention to values
        values = torch.matmul(dropout_attention, v)

        # Compute additional distance-aware term for values
        if self.embed_values:
            rv = self.embed_v(rel_dists)
            rv = rv.unsqueeze(-4)  # [Batch, 1, SeqLen, SeqLen, Dims]
            values = values + torch.einsum(
                "bhqrd,bhqrd->bhqd", dropout_attention.unsqueeze(-1), rv
            )

        # Unify heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)

        if need_weights:
            return values, attention
        return (values,)
