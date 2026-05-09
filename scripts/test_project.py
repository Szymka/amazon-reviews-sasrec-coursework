from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.seqrec.dataset import build_category_datasets
from models.seqrec.model import SASRec
from evaluation.metrics import evaluate, ndcg_at_k, hit_rate_at_k


def test_data_loading():
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)
    
    try:
        datasets = build_category_datasets(
            processed_root="examples",
            category="tiny_sample",
            maxlen=10
        )
        
        print(f"✓ 数据集加载成功")
        print(f"  - 训练集: {len(datasets['train'])} 样本")
        print(f"  - 验证集: {len(datasets['dev'])} 样本")
        print(f"  - 测试集: {len(datasets['test'])} 样本")
        
        sample = datasets['train'][0]
        print(f"  - 样本示例: user_id={sample['user_id']}, target_id={sample['target_id']}")
        print(f"  - 序列长度: {sample['seq_len']}, padding后: {len(sample['input_ids'])}")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def test_model():
    print("\n" + "=" * 60)
    print("测试 2: 模型创建和前向传播")
    print("=" * 60)
    
    try:
        model = SASRec(
            num_items=8,
            maxlen=10,
            hidden_units=64,
            num_blocks=2,
            num_heads=2,
            dropout_rate=0.2,
            padding_id=0
        )
        
        print(f"✓ 模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters())}")
        
        input_ids = torch.randint(0, 8, (2, 10))
        logits = model.predict(input_ids)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {input_ids.shape}")
        print(f"  - 输出形状: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    print("\n" + "=" * 60)
    print("测试 3: NDCG@10 评估函数")
    print("=" * 60)
    
    try:
        batch_size = 4
        num_items = 10
        
        predicted = torch.randn(batch_size, num_items)
        targets = torch.tensor([1, 3, 5, 7])
        
        ndcg = ndcg_at_k(predicted, targets, k=10)
        hr = hit_rate_at_k(predicted, targets, k=10)
        
        print(f"✓ NDCG@10 计算成功: {ndcg:.4f}")
        print(f"✓ HitRate@10 计算成功: {hr:.4f}")
        
        metrics = evaluate(predicted, targets, k=10)
        print(f"✓ 完整评估指标:")
        for key, value in metrics.items():
            print(f"  - {key}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 评估函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    print("\n" + "=" * 60)
    print("测试 4: 配置文件加载")
    print("=" * 60)
    
    try:
        from models.seqrec.dataset import load_simple_yaml
        
        config_path = Path(__file__).parent.parent / "configs" / "seqrec_industrial.yaml"
        config = load_simple_yaml(str(config_path))
        
        print(f"✓ 配置文件加载成功")
        print(f"  - category: {config.get('category')}")
        print(f"  - maxlen: {config.get('maxlen')}")
        print(f"  - hidden_units: {config.get('hidden_units')}")
        print(f"  - num_blocks: {config.get('num_blocks')}")
        print(f"  - num_heads: {config.get('num_heads')}")
        print(f"  - dropout_rate: {config.get('dropout_rate')}")
        print(f"  - early_stop_patience: {config.get('early_stop_patience')}")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def main():
    print("\n开始测试 SASRec 项目...\n")
    
    results = []
    
    results.append(("数据加载", test_data_loading()))
    results.append(("模型功能", test_model()))
    results.append(("评估函数", test_metrics()))
    results.append(("配置加载", test_config_loading()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！项目代码无误。")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
