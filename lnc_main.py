from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lnc_model import LncFeatureExtractor


DATA_DIR = Path(__file__).resolve().parent / "data" / "train" / "gfp"
DEFAULT_CFG = {
    "index": str(DATA_DIR / "train_lncRNAindex.txt"),
    "fasta": str(DATA_DIR / "train_lncRNA_seq.fa"),
    "coexp": str(DATA_DIR / "train_lncRNA_coexpression_sim.csv"),
    "lp": str(DATA_DIR / "train_lncRNA-protein-sim.csv"),
    "lm": str(DATA_DIR / "train_lncRNA-miRNA-sim.csv"),
    "out_npy": str(DATA_DIR / "train_vae.npy"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "batch": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "seed": 42,
    "lnc_dim": 256,
    "vae_beta": 1.0,
    "dropout": 0.1,
    "kmer": 4,
    "seq_nhead": 8,
    "seq_layers": 4,
    "seq_dim_ff": 512,
    "seq_dropout": 0.1,
    "knn_k": 15,
    "min_epochs": 30,
    "patience": 15,
    "min_delta": 1e-4,
    "min_delta_rel": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export lncRNA features")
    for name, default in DEFAULT_CFG.items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_index_names(path: str) -> list[str]:
    names = []
    with open(path, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected an index and an identifier")
            names.append(parts[1])
    if not names:
        raise ValueError(f"Index is empty: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Index contains duplicate identifiers: {path}")
    return names


def read_fasta(path: str) -> dict[str, str]:
    records = {}
    gene_sequences = {}
    aliases = {}
    header = None
    chunks = []

    def commit() -> None:
        if header is None:
            return
        seq = "".join(chunks).upper().replace("U", "T")
        if not seq:
            raise ValueError(f"Empty FASTA sequence for {header!r} in {path}")
        ids = [part.rstrip(">") for part in header.split("|") if part.rstrip(">")]
        gene = ids[1] if len(ids) == 2 else header
        gene_sequences.setdefault(gene, seq)
        for identifier in ids:
            aliases[identifier] = gene
        records.setdefault(header, seq)

    with open(path, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"{path}:{line_number}: FASTA header is empty")
                header = fields[0]
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence precedes FASTA header")
                chunks.append("".join(line.split()))
    commit()
    records.update(gene_sequences)
    records.update({identifier: gene_sequences[gene] for identifier, gene in aliases.items()})
    if not records:
        raise ValueError(f"FASTA contains no sequences: {path}")
    return records


def load_csv_matrix(path: str, count: int) -> np.ndarray:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [[float(value) for value in row] for row in csv.reader(handle) if row]
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (count, count):
        raise ValueError(f"Expected {(count, count)} in {path}, received {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Matrix contains NaN or Inf: {path}")
    return matrix


def standardize_rows(matrix: np.ndarray) -> np.ndarray:
    values = matrix.astype(np.float64)
    standardized = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-8)
    result = standardized.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Standardized matrix contains NaN or Inf")
    return result


def sim_recon_loss(h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(h, p=2, dim=-1)
    prediction = normalized @ normalized.t()
    target = target.clamp(-1.0, 1.0)
    target = 0.5 * (target + target.t())
    return F.mse_loss(prediction, target)


def validate_config(cfg: dict) -> None:
    for name in ("epochs", "batch", "lnc_dim", "kmer", "seq_nhead", "seq_layers", "seq_dim_ff", "knn_k", "min_epochs", "patience"):
        if cfg[name] < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("lr", "grad_clip"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "vae_beta", "min_delta", "min_delta_rel"):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("dropout", "seq_dropout"):
        if not np.isfinite(cfg[name]) or not 0 <= cfg[name] < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if cfg["lnc_dim"] % cfg["seq_nhead"]:
        raise ValueError("lnc_dim must be divisible by seq_nhead")


@torch.no_grad()
def export_features(
    model: LncFeatureExtractor,
    sequences: list[str],
    views: tuple[torch.Tensor, ...],
    cfg: dict,
) -> np.ndarray:
    model.eval()
    set_seed(cfg["seed"])
    seq_features = torch.cat([
        model.seq_encoder(sequences[start:start + cfg["batch"]])
        for start in range(0, len(sequences), cfg["batch"])
    ])
    h, _ = model.lnc_branch(seq_features, *views, knn_k=cfg["knn_k"])
    features = h.detach().float().cpu().numpy()
    expected = (len(sequences), cfg["lnc_dim"])
    if features.shape != expected or not np.isfinite(features).all():
        raise RuntimeError(f"Export must be finite and have shape {expected}; received {features.shape}")
    output = Path(cfg["out_npy"])
    if output.suffix.lower() != ".npy":
        raise ValueError("out_npy must have a .npy extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    print(f"[OK] Saved {output}, shape={features.shape}")
    return features


def train(cfg: dict) -> np.ndarray:
    cfg = {**DEFAULT_CFG, **cfg}
    validate_config(cfg)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    names = read_index_names(cfg["index"])
    fasta = read_fasta(cfg["fasta"])
    missing = [name for name in names if name not in fasta]
    if missing:
        raise ValueError(f"FASTA is missing {len(missing)} indexed identifiers: {missing[:10]}")
    sequences = [fasta[name] for name in names]
    count = len(names)
    views = tuple(
        torch.as_tensor(standardize_rows(load_csv_matrix(cfg[key], count)), device=device)
        for key in ("coexp", "lp", "lm")
    )
    model = LncFeatureExtractor(
        num_lnc=count,
        dim=cfg["lnc_dim"],
        kmer=cfg["kmer"],
        nhead=cfg["seq_nhead"],
        num_layers=cfg["seq_layers"],
        dim_feedforward=cfg["seq_dim_ff"],
        dropout=cfg["dropout"],
        seq_dropout=cfg["seq_dropout"],
        beta=cfg["vae_beta"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    order = np.arange(count)
    best_loss = None
    best_epoch = 0
    bad_count = 0
    print(f"[TRAIN] N={count}, device={device}, epochs={cfg['epochs']}, batch={cfg['batch']}")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        np.random.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, count, cfg["batch"]):
            batch_idx = order[start:start + cfg["batch"]]
            indices = torch.as_tensor(batch_idx, device=device, dtype=torch.long)
            batch_views = tuple(view.index_select(0, indices) for view in views)
            batch_seqs = [sequences[int(i)] for i in batch_idx]
            optimizer.zero_grad(set_to_none=True)
            h, aux = model(batch_seqs, *batch_views, knn_k=cfg["knn_k"])
            loss_sim = sum(sim_recon_loss(h, view.index_select(1, indices)) for view in batch_views)
            loss = aux["vae_loss"] + loss_sim
            if not torch.isfinite(h).all() or not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite features or loss at epoch={epoch}, step={steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach().item()
            steps += 1
        average_loss = loss_sum / steps
        if epoch == 1 or epoch % 10 == 0:
            print(f"[TRAIN] epoch={epoch:04d}/{cfg['epochs']}, loss={average_loss:.6f}")
        improved = best_loss is None
        if best_loss is not None:
            improvement = best_loss - average_loss
            improved = improvement > cfg["min_delta"] and improvement / max(abs(best_loss), 1e-12) > cfg["min_delta_rel"]
        if improved:
            best_loss = average_loss
            best_epoch = epoch
            bad_count = 0
        elif epoch >= cfg["min_epochs"]:
            bad_count += 1
        if epoch >= cfg["min_epochs"] and bad_count >= cfg["patience"]:
            print(f"[EARLY_STOP] epoch={epoch}, best_epoch={best_epoch}, best_loss={best_loss:.6f}")
            break
    return export_features(model, sequences, views, cfg)


def main() -> None:
    train(vars(parse_args()))


if __name__ == "__main__":
    main()
