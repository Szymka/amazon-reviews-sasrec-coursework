from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.metrics import evaluate
from models.sasrec.dataset import DEFAULT_CATEGORIES, build_category_datasets


class SASRecDataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'target_id': torch.tensor(sample['target_id'], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SASRec model Top-K recommendations")
    parser.add_argument(
        '--category',
        type=str,
        default='Industrial_and_Scientific',
        choices=list(DEFAULT_CATEGORIES),
        help=f"Dataset category. Choices: {', '.join(DEFAULT_CATEGORIES)}",
    )
    parser.add_argument(
        '--processed-root',
        type=Path,
        default=Path('data/processed'),
        help="Root directory for processed data",
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        '--maxlen',
        type=int,
        default=50,
        help="Maximum sequence length",
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help="Top-K for evaluation",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help="Batch size for evaluation",
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
    
    print(f"Loading dataset: {args.category}")
    datasets = build_category_datasets(args.processed_root, args.category, args.maxlen)
    
    stats_file = args.processed_root / args.category / 'stats.json'
    stats = json.loads(stats_file.read_text(encoding='utf-8'))
    num_items = int(stats['num_items'])
    padding_id = int(stats.get('padding_id', 0))
    
    test_dataset = SASRecDataset(datasets['test'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    from models.sasrec.model import SASRec
    model = SASRec(
        num_items=num_items,
        maxlen=args.maxlen,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        padding_id=padding_id,
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(args.device)
            target_id = batch['target_id'].to(args.device)
            
            logits = model.predict(input_ids)
            
            all_logits.append(logits.cpu())
            all_targets.append(target_id.cpu())
    
    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    metrics = evaluate(logits_tensor, targets_tensor, k=args.topk)
    
    print(f"\nTest Results for {args.category}:")
    print(f"  NDCG@{args.topk}: {metrics['ndcg']:.4f}")
    print(f"  HitRate@{args.topk}: {metrics['hit_rate']:.4f}")
    print(f"  Recall@{args.topk}: {metrics['recall']:.4f}")
    print(f"  MRR@{args.topk}: {metrics['mrr']:.4f}")
    print(f"  Precision@{args.topk}: {metrics['precision']:.4f}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
