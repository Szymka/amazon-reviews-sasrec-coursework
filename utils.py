import os
from datetime import datetime
from pytz import timezone
from collections import defaultdict

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

    
def get_now_time_str():
    # 保持原有命名的习惯
    return datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')

# --- 以下是为你新增的数据加载逻辑 ---

def data_load(dataset_name):
    """
    从 data/ 目录下读取数据集文件。
    返回: User (字典), usernum (用户总数), itemnum (物品总数)
    """
    # 自动定位路径，假设数据在根目录的 data 文件夹下
    path = f'data/amazon/{dataset_name}.txt'
    if not os.path.exists(path):
        # 尝试不同后缀
        path = f'data/amazon/{dataset_name}' if dataset_name.endswith('.txt') else f'data/{dataset_name}.txt'
    
    if not os.path.exists(path):
        raise FileNotFoundError(f" 找不到数据文件: {path}")

    User = defaultdict(list)
    usernum = 0
    itemnum = 0
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 假设格式为: user_id item_id
            parts = line.split(' ')
            u, i = int(parts[0]), int(parts[1])
            
            User[u].append(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            
    return User, usernum, itemnum
    