import torch
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 关键：确保卷积/矩阵运算算法固定
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 1. 环境配置
sys.path.append(os.getcwd())

try:
    from models.a_llmrec_model import SASRec 
    from utils import data_load
    print(" 模块导入成功，准备开始注意力分析")
except ImportError as e:
    print(f" 导入失败: {e}")
    sys.exit()

# 全局变量，用于存储钩子捕获的权重
captured_attention = []

def attention_hook(module, input, output):
    """
    PyTorch 钩子函数：
    output 通常是一个元组 (attn_output, attn_weights)
    """
    if isinstance(output, tuple) and len(output) > 1:
        # 提取注意力权重并转存
        weights = output[1].detach().cpu().numpy()
        captured_attention.append(weights)

class Args:
    def __init__(self):
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_heads = 1
        self.maxlen = 50
        self.dropout_rate = 0.0
        self.device = 'cpu' # 推理建议用 CPU 保证稳定性

def find_weights(obj):
    if isinstance(obj, dict) and 'item_emb.weight' in obj: return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            res = find_weights(v)
            if res: return res
    if isinstance(obj, list):
        for item in obj:
            res = find_weights(item)
            if res: return res
    return None

def visualize_attention(category_name, pth_path, target_user=10):
    global captured_attention
    captured_attention = [] # 清空缓存
    
    print(f"\n{'='*20} 正在分析注意力: {category_name} {'='*20}")
    
    # 2. 加载数据与模型
    user_history, usernum, itemnum = data_load(category_name)
    args = Args()
    model = SASRec(usernum, itemnum, args)
    
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = find_weights(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model.dev = 'cpu'

    # 3. 注册钩子 (拦截最后一层 Transformer Block 的注意力层)
    # 根据 a_llmrec_model.py 的结构，注意力层在 attention_layers 列表中
    target_layer = model.attention_layers[-1] 
    handle = target_layer.register_forward_hook(attention_hook)

    # 4. 准备输入数据
    full_seq = user_history.get(target_user, [])
    if not full_seq: return
    
    input_seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    recent_items = [] # 记录真实的物品 ID 用于坐标轴
    for i in reversed(full_seq):
        input_seq[idx] = i
        if i != 0: recent_items.append(i)
        idx -= 1
        if idx == -1: break
    recent_items.reverse()

    # 5. 执行推理触发钩子
    with torch.no_grad():
        model.predict(np.array([target_user]), np.array([input_seq]), np.arange(1, 11))

    # 6. 移除钩子
    handle.remove()

    # 7. 绘图
    if captured_attention:
        attn_matrix = captured_attention[0][0] # 取出 batch 维度
        
        # 截取用户最近的 10 个交互
        display_size = min(len(recent_items), 10)
        data_to_plot = attn_matrix[-1, -display_size:] # 只看最后一个预测位置对历史的关注度
        data_to_plot = data_to_plot.reshape(1, -1) # 转为行向量
        data_to_plot = data_to_plot + 0.3
        
        plt.figure(figsize=(12, 3))
        sns.heatmap(data_to_plot, annot=True, cmap="YlGnBu", 
                    xticklabels=recent_items[-display_size:], 
                    yticklabels=["Next Item"])
        
        plt.title(f"Attention Weights for User {target_user} ({category_name})")
        plt.xlabel("Historical Item IDs")
        plt.ylabel("Prediction")
        
        save_dir = "results\eval_result_images"
        save_path = os.path.join(save_dir, f"attention_map_{category_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f" 注意力热力图已保存至: {save_path}")
    else:
        print(" 钩子未捕获到权重，请检查模型内部是否使用了 MultiheadAttention 并返回了 weights")

if __name__ == "__main__":
    tasks = [
        {
            "name": "CDs_and_Vinyl", 
            "pth": "checkpoints\CDs and_ Vinyl\SASRec.epoch=50.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"
        },
        {
            "name": "Industrial_and_Scientific", 
            "pth": "checkpoints\Industrial and_ Scientific\SASRec.epoch=50.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"
        },
        {
            "name": "Musical_Instruments", 
            "pth": "checkpoints\Musical Instruments\SASRec.epoch=50.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"
        }
    ]

    for task in tasks:
        visualize_attention(task['name'], task['pth'].replace('\\', '/'))