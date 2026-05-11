# Data Preprocess

## Leave-Last-Out 划分逻辑

对每个用户按交互时间升序排序，假设该用户共有 `N` 条交互：

- `train`：前 `N-2` 条交互
- `dev/valid`：第 `N-1` 条交互
- `test`：第 `N` 条交互

Amazon 官方 raw 文件名使用 `valid.csv.gz`。本项目 processed 阶段输出文件名使用 `dev.tsv`，即 `valid = dev = 验证集`。

## 输入文件

`scripts/preprocess_to_seqrec.py` 读取：

```text
data/raw/<category>/<category>.train.csv.gz
data/raw/<category>/<category>.valid.csv.gz
data/raw/<category>/<category>.test.csv.gz
```

每个 CSV 至少需要字段：

```text
user_id,parent_asin,rating,timestamp,history
```

`history` 是空格分隔的历史 `parent_asin` 序列。`NaN`、空字符串或 `None` 会被解析为空序列。

## 输出文件

每个类别输出到 `data/processed/<category>/`：

```text
train.tsv
dev.tsv
test.tsv
seqrec_sequence.txt
seqrec_interactions.txt
user2id.json
id2user.json
item2id.json
id2item.json
stats.json
```

`train.tsv` / `dev.tsv` / `test.tsv` 表头固定为：

```text
user_id_int	target_id	rating	timestamp	seq_ids	raw_user_id	raw_parent_asin
```

ID 规则：

- `user_id_int` 从 1 开始连续编号。
- `item_id_int` 从 1 开始连续编号。
- `0` 保留给 padding。
- 同一类别内 `train/dev/test` 共用一套 `user2id.json` 和 `item2id.json`。
- 不同类别之间不共享 ID 映射。

## 命令示例

检查 raw：

```powershell
python scripts/check_raw_data.py
```

单类别 dry-run：

```powershell
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific --dry-run
```

单类别预处理：

```powershell
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific --overwrite
```

三类别预处理：

```powershell
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific --overwrite
python scripts/preprocess_to_seqrec.py --categories Musical_Instruments --overwrite
python scripts/preprocess_to_seqrec.py --categories CDs_and_Vinyl --overwrite
```

检查 processed：

```powershell
python scripts/check_processed_data.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl
```

Windows PowerShell 如果找不到 `python`，使用：

```powershell
$env:PYTHONIOENCODING='utf-8'; & "C:\Path\To\Your\Python\python.exe" scripts/check_processed_data.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl
```

## 注意

不要把 `data/raw/` 和 `data/processed/` 下的完整数据提交到 GitHub。GitHub 仓库只保留代码、文档、配置、小样例和结果摘要。
