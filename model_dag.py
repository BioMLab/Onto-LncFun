from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class TopoCausalRefiner(nn.Module):

    def __init__(self, dim: int, nhead: int=8, dropout: float=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, H: torch.Tensor, topo) -> torch.Tensor:
        device = H.device
        T, D = (H.size(0), H.size(1))
        if T == 0:
            return H
        if isinstance(topo, list):
            order = torch.tensor(topo, device=device, dtype=torch.long)
        else:
            order = topo.to(device)
        H_ord = H.index_select(0, order).unsqueeze(0)
        causal = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(H_ord, H_ord, H_ord, attn_mask=causal, need_weights=False)
        H1 = self.ln1(H_ord + attn_out)
        H2 = self.ln2(H1 + self.ffn(H1))
        inv = torch.empty_like(order)
        inv[order] = torch.arange(T, device=device)
        H_ref = H2.squeeze(0).index_select(0, inv)
        return H_ref

class GOEncoderRGCN(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_relations: int, dropout: float=0.1, act: str='tanh', residual: bool=True):
        super().__init__()
        self.residual = residual and in_dim == out_dim
        self.act = nn.Tanh() if act == 'tanh' else nn.GELU() if act == 'gelu' else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.rgcn1 = RGCNConv(in_dim, hid_dim, num_relations)
        self.rgcn2 = RGCNConv(hid_dim, out_dim, num_relations)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x0: torch.Tensor, triples_or_edges: torch.LongTensor) -> torch.Tensor:
        h, r, t = (triples_or_edges[:, 0], triples_or_edges[:, 1], triples_or_edges[:, 2])
        ei = torch.stack([h, t], dim=0)
        x = self.rgcn1(x0, ei, r)
        x = self.ln1(self.act(x))
        x = self.drop(x)
        x = self.rgcn2(x, ei, r)
        x = self.ln2(x)
        if self.residual:
            x = x + x0
        return x

def topological_sort(num_nodes: int, edge_index: torch.Tensor) -> List[int]:
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    device = edge_index.device
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    indeg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long, device=device))
    children = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        children[u].append(v)
    q = [int(i) for i in torch.nonzero(indeg == 0, as_tuple=True)[0].tolist()]
    topo = []
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        topo.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != num_nodes:
        raise RuntimeError('Graph is not a DAG (cycle detected).')
    return topo

def build_adj_lists(num_nodes: int, edge_index: torch.Tensor) -> Tuple[List[List[int]], List[List[int]]]:
    parents = [[] for _ in range(num_nodes)]
    children = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        children[u].append(v)
        parents[v].append(u)
    return (parents, children)

def compute_levels(num_nodes: int, edge_index: torch.Tensor, topo: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    device = edge_index.device
    parents = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        parents[v].append(u)
    levels = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for u in topo:
        if len(parents[u]) == 0:
            levels[u] = 0
        else:
            p = torch.tensor(parents[u], device=device)
            levels[u] = torch.max(levels.index_select(0, p)) + 1
    max_lv = int(levels.max().item()) if num_nodes > 0 else 0
    level_nodes = []
    for lv in range(max_lv + 1):
        idx = torch.nonzero(levels == lv, as_tuple=True)[0]
        level_nodes.append(idx)
    return (levels, level_nodes)

class LevelWiseTDUnit(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, 4 * dim)
        self.U = nn.Linear(dim, 4 * dim)

    def forward(self, x_level: torch.Tensor, h_p: torch.Tensor, c_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.W(x_level) + self.U(h_p)
        i, o, u, f = torch.chunk(gates, 4, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        f = torch.sigmoid(f)
        u = torch.tanh(u)
        c = i * u + f * c_p
        h = o * torch.tanh(c)
        return (c, h)

class LevelWiseBUUnit(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.ioux = nn.Linear(dim, 3 * dim)
        self.iouh = nn.Linear(dim, 3 * dim)
        self.fx = nn.Linear(dim, dim)
        self.fh = nn.Linear(dim, dim)

    def forward(self, x_all: torch.Tensor, H_all: torch.Tensor, C_all: torch.Tensor, edge_cp: torch.Tensor, parent_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_all.device
        T, D = (x_all.size(0), x_all.size(1))
        child = edge_cp[0]
        parent = edge_cp[1]
        pm = parent_mask[parent]
        child = child[pm]
        parent = parent[pm]
        lvl_nodes = torch.nonzero(parent_mask, as_tuple=True)[0]
        if lvl_nodes.numel() == 0:
            empty = x_all.new_zeros((0, D))
            return (empty, empty, lvl_nodes)
        h_sum = torch.zeros((T, D), device=device, dtype=x_all.dtype)
        if child.numel() > 0:
            h_sum.index_add_(0, parent, H_all.index_select(0, child))
        deg = torch.zeros((T,), device=device, dtype=x_all.dtype)
        if child.numel() > 0:
            deg.index_add_(0, parent, torch.ones_like(parent, dtype=x_all.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        h_bar = h_sum / deg
        iou_all = self.ioux(x_all) + self.iouh(h_bar)
        i_all, o_all, u_all = torch.chunk(iou_all, 3, dim=-1)
        i_all = torch.sigmoid(i_all)
        o_all = torch.sigmoid(o_all)
        u_all = torch.tanh(u_all)
        fc_all = torch.zeros((T, D), device=device, dtype=x_all.dtype)
        if child.numel() > 0:
            fx_parent = self.fx(x_all.index_select(0, parent))
            fh_child = self.fh(H_all.index_select(0, child))
            f_e = torch.sigmoid(fx_parent + fh_child)
            fc_all.index_add_(0, parent, f_e * C_all.index_select(0, child))
        c_all = i_all * u_all + fc_all
        h_all = o_all * torch.tanh(c_all)
        c_level = c_all.index_select(0, lvl_nodes)
        h_level = h_all.index_select(0, lvl_nodes)
        return (c_level, h_level, lvl_nodes)

class ModelB_DAG(nn.Module):

    def __init__(self, num_go, go_in_dim, lnc_in_dim=256, lnc_dim=256, go_dim=512, tl_dim=256, num_go_rel=8, inject_alpha=0.3, dropout=0.2):
        super().__init__()
        self.num_go = int(num_go)
        self.tl_dim = int(tl_dim)
        self.go_rgcn = GOEncoderRGCN(go_in_dim, go_dim, go_dim, num_go_rel)
        self.Wg = nn.Linear(go_dim, self.tl_dim, bias=False)
        self.xg_ln = nn.LayerNorm(self.tl_dim)
        self.external_lnc_proj = nn.Linear(lnc_in_dim, lnc_dim)
        self.Wl = nn.Linear(lnc_dim, self.tl_dim, bias=True)
        self.inject_attn = nn.MultiheadAttention(embed_dim=self.tl_dim, num_heads=min(8, self.tl_dim // 64 if self.tl_dim >= 64 else 1), batch_first=True)
        hidden = max(256, 2 * self.tl_dim)
        self.inject_mlp = nn.Sequential(nn.Linear(3 * self.tl_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.tl_dim))
        self.inject_alpha = float(inject_alpha)
        self.td_unit = LevelWiseTDUnit(self.tl_dim)
        self.td_attn_fc = nn.Linear(3 * self.tl_dim, 1)
        self.bu_unit = LevelWiseBUUnit(self.tl_dim)
        self.bifuse_gate = nn.Linear(2 * self.tl_dim, self.tl_dim)
        self.classifier_U = nn.Parameter(torch.empty(self.num_go, self.tl_dim))
        self.classifier_b = nn.Parameter(torch.zeros(self.num_go))
        nn.init.xavier_uniform_(self.classifier_U)
        self.post_refiner = TopoCausalRefiner(self.tl_dim, nhead=8, dropout=dropout)
        self._cache = {}

    def set_go_cache(self, go_biobert, go_triples_or_edges, edge_index):
        device = next(self.parameters()).device
        edge_index = edge_index.to(device)
        topo = topological_sort(self.num_go, edge_index)
        _, level_nodes = compute_levels(self.num_go, edge_index, topo)
        self._cache = {'edge': edge_index, 'topo': topo, 'level_nodes': level_nodes, 'go_biobert': go_biobert.to(device), 'go_triples_all': go_triples_or_edges.to(device)}

    def _compute_xg(self):
        return self.xg_ln(self.Wg(self.go_rgcn(self._cache['go_biobert'], self._cache['go_triples_all'])))

    def _edge_softmax_per_child(self, scores: torch.Tensor, child_index: torch.Tensor, T: int) -> torch.Tensor:
        device = scores.device
        if hasattr(torch.Tensor, 'scatter_reduce_'):
            max_buf = torch.full((T,), -1000000000.0, device=device, dtype=scores.dtype)
            max_buf.scatter_reduce_(0, child_index, scores, reduce='amax', include_self=True)
            exp = torch.exp(scores - max_buf.index_select(0, child_index))
            denom = torch.zeros((T,), device=device, dtype=scores.dtype)
            denom.index_add_(0, child_index, exp)
            denom = denom.index_select(0, child_index).clamp(min=1e-12)
            return exp / denom
        attn = torch.zeros_like(scores)
        uniq = torch.unique(child_index)
        for c in uniq.tolist():
            mask = child_index == c
            s = scores[mask]
            m = torch.max(s)
            e = torch.exp(s - m)
            attn[mask] = e / e.sum().clamp(min=1e-12)
        return attn

    def _dag_td_pass_levelwise(self, x: torch.Tensor, edge_pc: torch.Tensor, level_nodes: List[torch.Tensor], wl_proj_b: torch.Tensor) -> torch.Tensor:
        T, D = (x.size(0), x.size(1))
        device = x.device
        H = x.new_zeros(T, D)
        C = x.new_zeros(T, D)
        parent = edge_pc[0]
        child = edge_pc[1]
        wl_expand_all = wl_proj_b.unsqueeze(0).expand(T, -1)
        for nodes in level_nodes:
            if nodes.numel() == 0:
                continue
            mask_child = torch.zeros((T,), device=device, dtype=torch.bool)
            mask_child[nodes] = True
            em = mask_child[child]
            p = parent[em]
            c = child[em]
            if p.numel() == 0:
                x_level = x.index_select(0, nodes)
                h_p_lvl = x_level.new_zeros(x_level.size())
                c_p_lvl = x_level.new_zeros(x_level.size())
                c_lvl, h_lvl = self.td_unit(x_level, h_p_lvl, c_p_lvl)
                H[nodes] = h_lvl
                C[nodes] = c_lvl
                continue
            Hp = H.index_select(0, p)
            Cp = C.index_select(0, p)
            Xc = x.index_select(0, c)
            Wc = wl_expand_all.index_select(0, c)
            score_in = torch.cat([Hp, Xc, Wc], dim=-1)
            scores = self.td_attn_fc(score_in).squeeze(-1)
            attn = self._edge_softmax_per_child(scores, c, T).unsqueeze(-1)
            h_sum = torch.zeros((T, D), device=device, dtype=x.dtype)
            c_sum = torch.zeros((T, D), device=device, dtype=x.dtype)
            h_sum.index_add_(0, c, Hp * attn)
            c_sum.index_add_(0, c, Cp * attn)
            x_level = x.index_select(0, nodes)
            h_p_lvl = h_sum.index_select(0, nodes)
            c_p_lvl = c_sum.index_select(0, nodes)
            c_lvl, h_lvl = self.td_unit(x_level, h_p_lvl, c_p_lvl)
            H[nodes] = h_lvl
            C[nodes] = c_lvl
        return H

    def _dag_bu_pass_levelwise(self, x: torch.Tensor, edge_pc: torch.Tensor, level_nodes: List[torch.Tensor]) -> torch.Tensor:
        T, D = (x.size(0), x.size(1))
        device = x.device
        H = x.new_zeros(T, D)
        C = x.new_zeros(T, D)
        edge_cp = torch.stack([edge_pc[1], edge_pc[0]], dim=0)
        for nodes in reversed(level_nodes):
            if nodes.numel() == 0:
                continue
            parent_mask = torch.zeros((T,), device=device, dtype=torch.bool)
            parent_mask[nodes] = True
            c_lvl, h_lvl, lvl_nodes = self.bu_unit(x, H, C, edge_cp, parent_mask)
            if lvl_nodes.numel() > 0:
                H[lvl_nodes] = h_lvl
                C[lvl_nodes] = c_lvl
        return H

    def _bifuse(self, H_td, H_bu):
        g = torch.sigmoid(self.bifuse_gate(torch.cat([H_td, H_bu], dim=-1)))
        return g * H_td + (1.0 - g) * H_bu

    def _bi_dag_logits_one(self, xg: torch.Tensor, wl_proj_b: torch.Tensor, edge_pc: torch.Tensor, level_nodes: List[torch.Tensor]) -> torch.Tensor:
        T = xg.size(0)
        D = xg.size(1)
        wl_expand = wl_proj_b.unsqueeze(0).expand(T, -1)
        q = wl_proj_b.view(1, 1, D)
        k = xg.view(1, T, D)
        v = xg.view(1, T, D)
        ctx, _ = self.inject_attn(q, k, v)
        ctx_vec = ctx.view(1, D).expand(T, -1)
        x_in = torch.cat([xg, wl_expand, ctx_vec], dim=-1)
        x_inj = self.inject_mlp(x_in)
        alpha = self.inject_alpha
        if alpha <= 0.0:
            x = xg
        elif alpha >= 1.0:
            x = x_inj
        else:
            x = alpha * x_inj + (1.0 - alpha) * xg
        H_td = self._dag_td_pass_levelwise(x, edge_pc, level_nodes, wl_proj_b)
        H_bu = self._dag_bu_pass_levelwise(x, edge_pc, level_nodes)
        H = self._bifuse(H_td, H_bu)
        topo = self._cache.get('topo', None)
        if topo is None:
            raise RuntimeError('GO topological order is missing')
        H = self.post_refiner(H, topo)
        logits = (H * self.classifier_U).sum(dim=-1) + self.classifier_b
        return logits

    def forward(self, features):
        xg = self._compute_xg()
        wl = self.Wl(self.external_lnc_proj(features))
        logits = [self._bi_dag_logits_one(xg, item, self._cache['edge'], self._cache['level_nodes']) for item in wl]
        return torch.stack(logits) if logits else wl.new_zeros((0, self.num_go))
