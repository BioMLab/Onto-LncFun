from __future__ import annotations

import math
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class KmerTransformerEncoder(nn.Module):
    def __init__(
        self,
        k: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        if min(k, d_model, nhead, num_layers, dim_feedforward, max_len) < 1:
            raise ValueError("Transformer dimensions, k and max_len must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.k = k
        self.max_len = max_len
        self.vocab = {"".join(token): i for i, token in enumerate(product("ACGT", repeat=k))}
        self.unk_id = len(self.vocab)
        self.embed = nn.Embedding(self.unk_id + 1, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def tokenize_batch(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not seqs:
            raise ValueError("Sequence batch must not be empty")
        token_ids = []
        for seq in seqs:
            seq = seq.upper().replace("U", "T")
            count = min(max(1, len(seq) - self.k + 1), self.max_len)
            ids = [self.vocab.get(seq[i:i + self.k], self.unk_id) for i in range(count)]
            token_ids.append(torch.tensor(ids, dtype=torch.long))
        length = max(ids.numel() for ids in token_ids)
        padded = torch.full((len(seqs), length), self.unk_id, dtype=torch.long)
        pad_mask = torch.ones((len(seqs), length), dtype=torch.bool)
        for i, ids in enumerate(token_ids):
            padded[i, :ids.numel()] = ids
            pad_mask[i, :ids.numel()] = False
        return padded, pad_mask

    def forward(self, seqs: list[str]) -> torch.Tensor:
        tokens, pad_mask = self.tokenize_batch(seqs)
        tokens = tokens.to(self.embed.weight.device)
        pad_mask = pad_mask.to(self.embed.weight.device)
        x = self.encoder(self.pe(self.embed(tokens)), src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1)
        pooled = x.masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.norm(pooled)


class MultiViewVAE(nn.Module):
    def __init__(
        self,
        d_seq: int,
        d_c: int,
        d_lp: int,
        d_lm: int,
        d_proj: int = 256,
        d_latent: int = 256,
        hidden: int = 1024,
        beta: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.projections = nn.ModuleList(nn.Linear(d, d_proj) for d in (d_seq, d_c, d_lp, d_lm))
        self.view_norms = nn.ModuleList(nn.LayerNorm(d_proj) for _ in range(4))
        self.cat_norm = nn.LayerNorm(4 * d_proj)
        self.enc = nn.Sequential(
            nn.Linear(4 * d_proj, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, d_latent)
        self.logvar_head = nn.Linear(hidden, d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 4 * d_proj),
        )

    def project_views(self, *views: torch.Tensor) -> torch.Tensor:
        projected = [norm(F.relu(proj(x))) for x, proj, norm in zip(views, self.projections, self.view_norms)]
        return self.cat_norm(torch.cat(projected, dim=-1))

    def forward(
        self, x_seq: torch.Tensor, x_c: torch.Tensor, x_lp: torch.Tensor, x_lm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project_views(x_seq, x_c, x_lp, x_lm)
        encoded = self.enc(projected)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded).clamp(-8.0, 8.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.dec(z)
        recon_loss = F.mse_loss(recon, projected.detach())
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
        return z, recon_loss + self.beta * kl


@torch.no_grad()
def build_knn_graph_from_embeddings(z: torch.Tensor, k: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    if k < 1:
        raise ValueError("knn_k must be positive")
    n = z.size(0)
    if n <= 1:
        return torch.empty((2, 0), device=z.device, dtype=torch.long), z.new_empty(0)
    normalized = F.normalize(z, p=2, dim=-1)
    sim = normalized @ normalized.t()
    sim.fill_diagonal_(-torch.inf)
    k_eff = min(k, n - 1)
    weights, neighbors = sim.topk(k_eff, dim=1)
    rows = torch.arange(n, device=z.device).unsqueeze(1).expand(-1, k_eff)
    return torch.stack([rows.reshape(-1), neighbors.reshape(-1)]), weights.reshape(-1)


class LncRNAEncoderGCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def propagate(x: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        nodes = torch.arange(n, device=x.device)
        edges = torch.cat([edges, torch.stack([nodes, nodes])], dim=1)
        weights = torch.cat([weights, x.new_ones(n)])
        src, dst = edges
        degree = x.new_zeros(n).scatter_add_(0, dst, weights).clamp(min=1.0)
        inverse = degree.rsqrt()
        normalized = inverse[src] * weights * inverse[dst]
        output = torch.zeros_like(x)
        output.index_add_(0, dst, x[src] * normalized.unsqueeze(-1))
        return output

    def forward(self, x: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.lin1(self.propagate(x, edges, weights))))
        return self.lin2(self.propagate(h, edges, weights))


class LncBranch(nn.Module):
    def __init__(
        self,
        d_seq: int,
        d_c: int,
        d_lp: int,
        d_lm: int,
        dim: int = 256,
        beta: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mv_vae = MultiViewVAE(
            d_seq, d_c, d_lp, d_lm, d_proj=dim, d_latent=dim,
            hidden=max(512, dim * 4), beta=beta, dropout=dropout,
        )
        self.lnc_gcn = LncRNAEncoderGCN(dim, dim, dim, dropout)
        self.lnc_fuse_proj = nn.Linear(2 * dim, dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_c: torch.Tensor,
        x_lp: torch.Tensor,
        x_lm: torch.Tensor,
        knn_k: int = 15,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z, vae_loss = self.mv_vae(x_seq, x_c, x_lp, x_lm)
        edges, weights = build_knn_graph_from_embeddings(z, knn_k)
        h_gcn = self.lnc_gcn(z, edges, weights)
        h = self.lnc_fuse_proj(torch.cat([z, h_gcn], dim=-1))
        return h, {"z": z, "vae_loss": vae_loss, "edge_index": edges, "edge_weight": weights}


class LncFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_lnc: int,
        dim: int = 256,
        kmer: int = 4,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_dropout: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.seq_encoder = KmerTransformerEncoder(
            kmer, dim, nhead, num_layers, dim_feedforward, seq_dropout,
        )
        self.lnc_branch = LncBranch(dim, num_lnc, num_lnc, num_lnc, dim, beta, dropout)

    def forward(
        self,
        seqs: list[str],
        x_c: torch.Tensor,
        x_lp: torch.Tensor,
        x_lm: torch.Tensor,
        knn_k: int = 15,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.lnc_branch(self.seq_encoder(seqs), x_c, x_lp, x_lm, knn_k)
