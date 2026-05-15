import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义你上传的文件名
files = {
    "CDs_and_Vinyl": "results/coursework/CDs_and_Vinyl_metrics.jsonl",
    "Industrial_and_Scientific": "results/coursework/Industrial_and_Scientific_metrics.jsonl",
    "Musical_Instruments": "results/coursework/Musical_Instruments_metrics.jsonl"
}

os.makedirs("results\eval_result_images", exist_ok=True)

def plot_ndcg(name, path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # 过滤掉 source 标记，只保留 json 部分
            line = line.strip()
            if line.startswith('{'):
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    # 如果 epoch 有重复（如你上传的文件），取最后一次出现的记录
    df = df.drop_duplicates(subset=['epoch'], keep='last').sort_values('epoch')

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['valid_ndcg10'], marker='o', label='Valid NDCG@10')
    plt.plot(df['epoch'], df['test_ndcg10'], marker='s', label='Test NDCG@10', linestyle='--')
    
    plt.title(f'NDCG Convergence: {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results\eval_result_images\{name}_ndcg_curve.png")
    plt.close()
    print(f" {name} 的 NDCG 曲线已保存")

for name, path in files.items():
    plot_ndcg(name, path)










