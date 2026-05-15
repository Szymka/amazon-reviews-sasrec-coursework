import torch
import numpy as np
import os
import sys
import logging

# 配置日志
log_dir = "evaluation/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "top10_prediction.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'), # 记录到文件
        logging.StreamHandler(sys.stdout)               # 同时输出到屏幕
    ]
)
# 1. 环境配置
sys.path.append(os.getcwd())

try:
    from models.a_llmrec_model import SASRec 
    from utils import data_load
    logging.info(" 模块导入成功！")
except ImportError as e:
    logging.info(f" 导入失败: {e}")
    sys.exit()

# 2. 超参数配置
class Args:
    def __init__(self):
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_heads = 1
        self.maxlen = 50
        self.dropout_rate = 0.0
        # 【建议】：大作业推理任务直接用 CPU，避开显卡驱动和张量冲突
        self.device = 'cpu' 

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
    if hasattr(obj, 'state_dict'):
        try:
            sd = obj.state_dict(); 
            if 'item_emb.weight' in sd: return sd
        except: pass
    return None

def get_top10_prediction(category_name, pth_path, test_user_id=10):
    logging.info(f"\n{'='*20} 正在评估: {category_name} {'='*20}")
    
    # 3. 加载数据
    try:
        user_history, usernum, itemnum = data_load(category_name)
    except Exception as e:
        logging.info(f"❌ 加载数据失败: {e}")
        return

    # 4. 初始化模型 (统一在 CPU 上)
    args = Args()
    model = SASRec(usernum, itemnum, args)
    
    if not os.path.exists(pth_path):
        logging.info(f" 找不到权重文件: {pth_path}")
        return

    # 5. 加载权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = find_weights(checkpoint)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            model.to(args.device)
            model.dev = args.device # 关键：同步内部设备标记
            logging.info(f" 权重对齐成功！当前推理设备: {args.device}")
        except Exception as e:
            logging.info(f" 权重加载失败: {e}")
            return
    else:
        logging.info(f" 无法解析权重参数")
        return

    model.eval()

    # 6. 构造输入 (关键修复：直接使用 Numpy，不提前转 Tensor)
    full_seq = user_history.get(test_user_id, [])
    if not full_seq: return
    
    input_seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    for i in reversed(full_seq):
        input_seq[idx] = i
        idx -= 1
        if idx == -1: break
    
    # 准备传给 predict 的参数，保持为 Numpy/原生类型
    # 这样模型内部的 torch.LongTensor(log_seqs).to(self.dev) 就不会报错
    user_ids = np.array([test_user_id])
    log_seqs = np.array([input_seq])
    item_indices = np.arange(1, itemnum + 1)

    # 7. 执行推理
    try:
        with torch.no_grad():
            # 这里传入的是 Numpy 数组
            predictions = -model.predict(user_ids, log_seqs, item_indices)
            # 获取第一行的预测结果
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # 如果结果是多维的，取第一行
            if len(predictions.shape) > 1:
                predictions = predictions[0]

        # 8. 排序输出
        top10_idx = predictions.argsort()[:10]
        # 注意：物品 ID 通常从 1 开始
        top10_items = item_indices[top10_idx]

        logging.info(f" 用户 ID: {test_user_id}")
        logging.info(f" Top-10 推荐物品 ID 列表: {top10_items.tolist()}")
    except Exception as e:
        logging.info(f" 推理发生致命错误: {e}")

if __name__ == "__main__":
    # 请根据你本地实际的 .pth 路径进行修改
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
        get_top10_prediction(task['name'], task['pth'])