from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.sasrec.dataset import (
    DEFAULT_CATEGORIES,
    build_category_datasets,
    load_simple_yaml,
)
from evaluation.metrics import evaluate
from models.sasrec.model import SASRec


class SASRecDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'target_id': torch.tensor(sample['target_id'], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SASRec Training Script")
    
    parser.add_argument(
        '--category',
        type=str,
        default='Industrial_and_Scientific',
        choices=list(DEFAULT_CATEGORIES),
        help=f"Dataset category. Choices: {', '.join(DEFAULT_CATEGORIES)}",
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help="Path to config file (overrides other parameters)",
    )
    
    parser.add_argument(
        '--processed-root',
        type=Path,
        default=Path('data/processed'),
        help="Root directory for processed data",
    )
    
    parser.add_argument(
        '--maxlen',
        type=int,
        default=50,
        help="Maximum sequence length",
    )
    parser.add_argument(
        '--hidden-units',
        type=int,
        default=64,
        help="Embedding/hidden dimension",
    )
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=2,
        help="Number of transformer blocks",
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=2,
        help="Number of attention heads",
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help="Top-K for evaluation",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=5,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        '--early-stop-min-delta',
        type=float,
        default=0.0,
        help="Early stopping minimum delta",
    )
    
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=Path('train'),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        '--save-best-only',
        action='store_true',
        default=True,
        help="Only save the best model",
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use (cpu or cuda)",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if args.config is not None:
        config = load_simple_yaml(args.config)
        args.category = str(config.get('category', args.category))
        args.maxlen = int(config.get('maxlen', args.maxlen))
        args.hidden_units = int(config.get('hidden_units', args.hidden_units))
        args.num_blocks = int(config.get('num_blocks', args.num_blocks))
        args.num_heads = int(config.get('num_heads', args.num_heads))
        args.dropout_rate = float(config.get('dropout_rate', args.dropout_rate))
        args.learning_rate = float(config.get('learning_rate', args.learning_rate))
        args.batch_size = int(config.get('batch_size', args.batch_size))
        args.num_epochs = int(config.get('num_epochs', args.num_epochs))
        args.topk = int(config.get('topk', args.topk))
        args.seed = int(config.get('seed', args.seed))
        args.early_stop_patience = int(config.get('early_stop_patience', args.early_stop_patience))
        args.early_stop_min_delta = float(config.get('early_stop_min_delta', args.early_stop_min_delta))
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    print(f"Loading dataset: {args.category}")
    datasets = build_category_datasets(args.processed_root, args.category, args.maxlen)
    
    stats_file = args.processed_root / args.category / 'stats.json'
    stats = json.loads(stats_file.read_text(encoding='utf-8'))
    num_items = int(stats['num_items'])
    padding_id = int(stats.get('padding_id', 0))
    
    train_dataset = SASRecDataset(datasets['train'])
    dev_dataset = SASRecDataset(datasets['dev'])
    test_dataset = SASRecDataset(datasets['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = SASRec(
        num_items=num_items,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        padding_id=padding_id,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    best_dev_ndcg = 0.0
    early_stop_count = 0
    best_epoch = 0
    
    print(f"Training started on {args.device}")
    print(f"Hyperparameters: hidden_units={args.hidden_units}, num_blocks={args.num_blocks}, num_heads={args.num_heads}, dropout_rate={args.dropout_rate}")
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_id = batch['target_id'].to(device)
            
            optimizer.zero_grad()
            
            logits = model.predict(input_ids)
            loss = criterion(logits, target_id)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
            train_count += input_ids.size(0)
        
        train_loss /= train_count
        
        model.eval()
        dev_loss = 0.0
        dev_count = 0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                target_id = batch['target_id'].to(device)
                
                logits = model.predict(input_ids)
                loss = criterion(logits, target_id)
                
                dev_loss += loss.item() * input_ids.size(0)
                dev_count += input_ids.size(0)
                
                all_logits.append(logits.cpu())
                all_targets.append(target_id.cpu())
        
        dev_loss /= dev_count
        dev_logits = torch.cat(all_logits)
        dev_targets = torch.cat(all_targets)
        dev_metrics = evaluate(dev_logits, dev_targets, k=args.topk)
        
        epoch_time = time.time() - start_time
        
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev NDCG@{args.topk}: {dev_metrics['ndcg']:.4f} | "
            f"Dev HR@{args.topk}: {dev_metrics['hit_rate']:.4f}"
        )
        
        if dev_metrics['ndcg'] > best_dev_ndcg + args.early_stop_min_delta:
            best_dev_ndcg = dev_metrics['ndcg']
            best_epoch = epoch + 1
            early_stop_count = 0
            
            model_path = args.save_dir / f'sasrec_{args.category}_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"  Best model saved to {model_path}")
            
            config_path = args.save_dir / f'sasrec_{args.category}_best_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'category': args.category,
                    'epoch': best_epoch,
                    'ndcg': best_dev_ndcg,
                    'hyperparameters': {
                        'maxlen': args.maxlen,
                        'hidden_units': args.hidden_units,
                        'num_blocks': args.num_blocks,
                        'num_heads': args.num_heads,
                        'dropout_rate': args.dropout_rate,
                        'learning_rate': args.learning_rate,
                        'batch_size': args.batch_size,
                        'seed': args.seed,
                    }
                }, f, indent=2)
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop_patience:
                print(f"Early stopping triggered after {early_stop_count} epochs without improvement")
                break
    
    print(f"\nLoading best model from epoch {best_epoch}")
    best_model_path = args.save_dir / f'sasrec_{args.category}_best.pth'
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    test_count = 0
    all_test_logits = []
    all_test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            target_id = batch['target_id'].to(device)
            
            logits = model.predict(input_ids)
            loss = criterion(logits, target_id)
            
            test_loss += loss.item() * input_ids.size(0)
            test_count += input_ids.size(0)
            
            all_test_logits.append(logits.cpu())
            all_test_targets.append(target_id.cpu())
    
    test_loss /= test_count
    test_logits = torch.cat(all_test_logits)
    test_targets = torch.cat(all_test_targets)
    test_metrics = evaluate(test_logits, test_targets, k=args.topk)
    
    print(f"\nTest Results for {args.category}:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  NDCG@{args.topk}: {test_metrics['ndcg']:.4f}")
    print(f"  HitRate@{args.topk}: {test_metrics['hit_rate']:.4f}")
    print(f"  Recall@{args.topk}: {test_metrics['recall']:.4f}")
    print(f"  MRR@{args.topk}: {test_metrics['mrr']:.4f}")
    print(f"  Precision@{args.topk}: {test_metrics['precision']:.4f}")
    
    results_data = {
        'category': args.category,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'hyperparameters': {
            'maxlen': args.maxlen,
            'hidden_units': args.hidden_units,
            'num_blocks': args.num_blocks,
            'num_heads': args.num_heads,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'topk': args.topk,
            'seed': args.seed,
        }
    }
    
    results_path = args.save_dir / f'sasrec_{args.category}_test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    results_tables_dir = Path('results/tables')
    results_tables_dir.mkdir(parents=True, exist_ok=True)
    tables_path = results_tables_dir / f'sasrec_{args.category}_test_results.json'
    with open(tables_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results also saved to {tables_path}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
