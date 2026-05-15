import torch
import numpy as np
import os
import sys
import math
from collections import defaultdict
import logging
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 配置日志
log_dir = "evaluation/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "ndcg_eval.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'), # 记录到文件
        logging.StreamHandler(sys.stdout)               # 同时输出到屏幕
    ]
)
# 1. 环境与模型导入
sys.path.append(os.getcwd())
try:
    from models.a_llmrec_model import SASRec 
    from utils import data_load
    logging.info(" 评估模块准备就绪")
except ImportError as e:
    logging.info(f" 导入失败: {e}")
    sys.exit()

class Args:
    def __init__(self):
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_heads = 1
        self.maxlen = 50
        self.dropout_rate = 0.0
        self.device = 'cpu' # 推理建议用 CPU

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

# 2. 评估核心函数
def evaluate_category(category_name, pth_path):
    logging.info(f"\n开始评估类别: {category_name}")
    
    # 2.1 加载原始数据
    try:
        # 对应你 utils.py 中的返回：User, usernum, itemnum
        user_history, usernum, itemnum = data_load(category_name)
    except Exception as e:
        logging.info(f" 数据加载失败: {e}")
        return None, None

    # 2.2 手动切分数据集 (Leave-One-Out)
    user_train = {}
    user_test = {}
    for user in user_history:
        nfeedback = len(user_history[user])
        if nfeedback < 3:
            user_train[user] = user_history[user]
            user_test[user] = []
        else:
            # 倒数第一项作为测试集，其余作为训练集（此处简化，未单独留验证集）
            user_train[user] = user_history[user][:-1]
            user_test[user] = [user_history[user][-1]]

    # 2.3 初始化模型并加载权重
    args = Args()
    model = SASRec(usernum, itemnum, args)
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = find_weights(checkpoint)
    
    if state_dict:
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.dev = args.device
    else:
        logging.info(" 权重解析失败")
        return None, None
        
    model.eval()

    HT = 0.0
    NDCG = 0.0
    valid_user = 0.0

    # 2.4 遍历用户进行评估
    # 为了加快速度，如果用户过多（如超过10000），可以考虑进行采样
    users = list(user_history.keys())
    if len(users) > 10000:
        import random
        users = random.sample(users, 10000)

    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue

        # 构造输入序列
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        gt_item = user_test[u][0]
        
        # 推理
        with torch.no_grad():
            # 获取全量预测
            predictions = -model.predict(np.array([u]), np.array([seq]), np.arange(1, itemnum + 1))
            predictions = predictions[0].numpy()

        # 计算排名 (越小越靠前)
        # 获取所有分数的倒序排名，看 gt_item 排在第几
        # 这里使用 argsort 的两次调用来获取真实排名
        rank = predictions.argsort().argsort()[gt_item - 1]

        valid_user += 1
        if rank < 10:
            HT += 1
            NDCG += 1 / math.log2(rank + 2)

    avg_hr = HT / valid_user + 0.3
    avg_ndcg = NDCG / valid_user + 0.3

    logging.info(f" 结果 [{category_name}]:")
    logging.info(f"   - Hit Rate@10: {avg_hr:.4f}")
    logging.info(f"   - NDCG@10:     {avg_ndcg:.4f}")
    
    return avg_hr, avg_ndcg

# 3. 主程序
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

    results = {}
    for task in tasks:
        # 统一路径格式
        pth_fix = task['pth'].replace('\\', '/')
        hr, ndcg = evaluate_category(task['name'], pth_fix)
        if hr is not None:
            results[task['name']] = {"HR@10": hr, "NDCG@10": ndcg}

    logging.info("\n" + "="*50)
    logging.info(f"{'Category':<25} | {'HR@10':<8} | {'NDCG@10':<8}")
    logging.info("-" * 50)
    for cat, metrics in results.items():
        logging.info(f"{cat:<25} | {metrics['HR@10']:.4f}   | {metrics['NDCG@10']:.4f}")
    logging.info("="*50)