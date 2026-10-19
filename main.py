import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from evaluation import evaluate_paper_metrics, load_ic_weights
from model_dag import ModelB_DAG
from utils import depth_cost, hierarchical_loss, load_features, read_names, read_samples, read_test_labels, save_predictions


ROOT = Path(__file__).resolve().parent
METRICS = ('Fmax', 'AUPR', 'U-Fmax', 'U-AUPR')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lnc_batch', type=int, default=32)
    parser.add_argument('--lnc_dim', type=int, default=256)
    parser.add_argument('--go_dim', type=int, default=512)
    parser.add_argument('--treelstm_hidden', type=int, default=256)
    parser.add_argument('--inject_alpha', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.lnc_batch < 1 or args.lr <= 0:
        parser.error('epochs, lnc_batch and lr must be positive')
    if args.treelstm_hidden < 8 or args.treelstm_hidden % 8 or min(args.lnc_dim, args.go_dim) < 1:
        parser.error('treelstm_hidden must be a positive multiple of 8; feature dimensions must be positive')
    heads = min(8, args.treelstm_hidden // 64 if args.treelstm_hidden >= 64 else 1)
    if args.treelstm_hidden % heads:
        parser.error('treelstm_hidden is incompatible with the cross-attention head count')
    return args


def load_go(data, device):
    folder = data / 'GO/pruned'
    names = read_names(folder / 'go_names.K.txt')
    features = torch.load(folder / 'go_biobert.K.pt', map_location=device, weights_only=True).float()
    triples = torch.as_tensor(np.load(folder / 'go_triples.K.npy', allow_pickle=False), dtype=torch.long, device=device)
    edges = torch.as_tensor(np.load(folder / 'go_tree_edges.K.npy', allow_pickle=False), dtype=torch.long, device=device)
    if edges.ndim != 2 or (edges.shape[0] != 2 and edges.shape[1] != 2):
        raise ValueError('Invalid GO hierarchy shape')
    if edges.shape[0] != 2:
        edges = edges.t().contiguous()
    if features.ndim != 2 or len(names) != features.shape[0] or not torch.isfinite(features).all():
        raise ValueError('GO features and identifiers are invalid or misaligned')
    if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
        raise ValueError('Invalid GO triples')
    if triples.min() < 0 or triples[:, [0, 2]].max() >= len(names) or (edges.numel() and (edges.min() < 0 or edges.max() >= len(names))):
        raise ValueError('GO graph index is out of bounds')
    weights, _ = load_ic_weights(names, data / 'GO/term2id_IC.txt', data / 'GO/go-basic.obo')
    return names, features, triples, edges, weights


def model_config(args, input_dim, go_features, triples, num_go):
    return {'num_go': num_go, 'go_in_dim': int(go_features.shape[1]), 'lnc_in_dim': input_dim,
            'lnc_dim': args.lnc_dim, 'go_dim': args.go_dim, 'tl_dim': args.treelstm_hidden,
            'num_go_rel': int(triples[:, 1].max()) + 1,
            'inject_alpha': args.inject_alpha, 'dropout': args.dropout}


@torch.no_grad()
def predict(model, features, batch_size, device):
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start:start + batch_size].to(device))
        probabilities = torch.sigmoid(logits)
        if not torch.isfinite(probabilities).all():
            raise ValueError('Predictions contain nonfinite values')
        chunks.append(probabilities.cpu().numpy())
    return np.concatenate(chunks)


def print_metrics(prefix, values):
    print(prefix + ' | ' + ' '.join(f'{name}={values[name]:.4f}' for name in METRICS), flush=True)


def export_evaluation(folder, stem, names, go_names, probabilities, labels, mask, weights):
    metrics = evaluate_paper_metrics(probabilities, labels, weights, mask)
    save_predictions(folder / f'{stem}.tsv', names, go_names, probabilities, labels, mask)
    (folder / f'{stem}.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print_metrics(stem, metrics)
    return metrics


def split_indices(names):
    indices = np.arange(len(names))
    np.random.shuffle(indices)
    val_count = max(1, round(len(indices) * 0.15))
    if val_count >= len(indices):
        raise ValueError('At least one training and validation lncRNA are required')
    return indices[val_count:], indices[:val_count]


def train(args, go, names, features, labels, mask):
    go_names, go_features, triples, edges, weights = go
    device = torch.device(args.device)
    config = model_config(args, features.shape[1], go_features, triples, len(go_names))
    model = ModelB_DAG(**config).to(device)
    model.set_go_cache(go_features, triples, edges)
    roots = [i for i, name in enumerate(go_names) if name in {'GO:0008150', 'GO:0005575', 'GO:0003674'}]
    costs = depth_cost(len(go_names), edges, roots, device)
    train_indices, val_indices = split_indices(names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = (-1.0, -1.0)
    checkpoint = args.output_dir / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices.copy()
        np.random.shuffle(order)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(order), args.lnc_batch):
            batch = order[start:start + args.lnc_batch]
            logits = model(features[batch].to(device))
            batch_labels = torch.as_tensor(labels[batch], device=device)
            batch_mask = torch.as_tensor(mask[batch], device=device)
            loss = hierarchical_loss(logits, batch_labels, batch_mask, costs, edges)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is nonfinite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        probabilities = predict(model, features[val_indices], args.lnc_batch, device)
        metrics = evaluate_paper_metrics(probabilities, labels[val_indices], weights, mask[val_indices])
        print_metrics(f'Epoch {epoch} Loss={total_loss / max(steps, 1):.6f}', metrics)
        score = (metrics['Fmax'], metrics['AUPR'])
        if score[0] > best[0] + 1e-12 or (abs(score[0] - best[0]) <= 1e-12 and score[1] > best[1]):
            best = score
            torch.save({'model_state': model.state_dict(), 'model_config': config,
                        'go_names': go_names, 'epoch': epoch, 'metrics': metrics,
                        'train_names': [names[i] for i in train_indices],
                        'val_names': [names[i] for i in val_indices]}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    print_metrics(f"Best validation model: epoch {payload['epoch']} | {checkpoint}", payload['metrics'])
    model.load_state_dict(payload['model_state'], strict=True)
    probabilities = predict(model, features[val_indices], args.lnc_batch, device)
    export_evaluation(args.output_dir, 'validation', [names[i] for i in val_indices], go_names,
                      probabilities, labels[val_indices], mask[val_indices], weights)
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    go = load_go(args.data_dir, device)
    go_names, go_features, triples, edges, weights = go
    train_dir = args.data_dir / 'train/gfp'
    names = read_names(train_dir / 'train_lncRNAindex.txt')
    features = load_features(train_dir / 'train_vae.npy', names)
    tests = {}
    for split in ('test55', 'test68'):
        folder = args.data_dir / 'test' / split
        test_names = read_names(folder / 'test_lncRNAindex.txt')
        overlap = set(names) & set(test_names)
        if overlap:
            raise ValueError(f'Training/test overlap in {split}: {sorted(overlap)}')
        test_features = load_features(folder / 'test_vae.npy', test_names)
        if test_features.shape[1] != features.shape[1]:
            raise ValueError('Training and test feature dimensions differ')
        test_labels, test_mask = read_test_labels(folder / 'test.expanded.txt', test_names, go_names)
        tests[split] = (test_names, test_features, test_labels, test_mask)
    if args.test_only:
        checkpoint = args.checkpoint or args.output_dir / 'best_model.pt'
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload['go_names'] != go_names or payload['model_config']['lnc_in_dim'] != features.shape[1]:
            raise ValueError('Checkpoint identifiers or feature dimensions differ from the dataset')
        for split, (test_names, _, _, _) in tests.items():
            if (set(payload['train_names']) | set(payload['val_names'])) & set(test_names):
                raise ValueError(f'Checkpoint contains test lncRNAs: {split}')
        model = ModelB_DAG(**payload['model_config']).to(device)
        model.load_state_dict(payload['model_state'], strict=True)
        model.set_go_cache(go_features, triples, edges)
    else:
        labels, mask = read_samples(train_dir / 'sample.txt', names, go_names)
        model = train(args, go, names, features, labels, mask)
    for split, (test_names, test_features, test_labels, test_mask) in tests.items():
        probabilities = predict(model, test_features, args.lnc_batch, device)
        export_evaluation(args.output_dir, split, test_names, go_names,
                          probabilities, test_labels, test_mask, weights)


if __name__ == '__main__':
    main()
