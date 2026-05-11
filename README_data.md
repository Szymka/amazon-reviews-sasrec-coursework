# Amazon Reviews 2023 数据预处理说明

## 1. 任务目标

本项目负责推荐系统大作业的第一部分：数据预处理。目标是将 Amazon Reviews 2023 的三个 5-Core 商品类别数据整理为序列推荐模型 SASRec 可读取的格式。

本部分只处理数据下载、清洗、时间序列划分、ID 映射、格式转换和数据检查，不负责模型训练、调参或最终指标计算。

## 2. 使用的数据类别

本次大作业使用以下三个 Amazon Reviews 2023 5-Core 类别：

1. `Industrial_and_Scientific`
2. `Musical_Instruments`
3. `CDs_and_Vinyl`

## 3. 数据划分原则

对每个用户按照交互时间升序排序，假设该用户共有 `N` 个交互：

- 训练集：前 `N-2` 个交互
- 验证集：第 `N-1` 个交互
- 测试集：第 `N` 个交互

因此，每个保留用户至少需要 3 个交互。

## 4. 下载和原始输入文件

原始数据建议放在 `data/raw/<category>/` 下，其中 `<category>` 为三个类别之一。

下载阶段每个类别的必需文件是官方 Leave-Last-Out 5-Core split：

- `<category>.train.csv.gz`
- `<category>.valid.csv.gz`
- `<category>.test.csv.gz`

其中官方文件名使用 `valid`。后续预处理阶段会把 `valid.csv.gz` 转换为处理后的 `dev.tsv`。

`review` 和 `meta` 原始文件只作为可选辅助数据，体积可能较大，默认不会下载。官方 raw 文件名不包含 `.review` / `.meta` 后缀；只有显式传入 `--include-review-meta` 时才会下载：

- `<category>.jsonl.gz`：User Review，官方命名
- `meta_<category>.jsonl.gz`：Item Metadata，官方命名

示例路径：

```text
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.train.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.valid.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.test.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.jsonl.gz
data/raw/Industrial_and_Scientific/meta_Industrial_and_Scientific.jsonl.gz
```

`train`、`valid`、`test` 是 SASRec 预处理必需文件；`review`、`meta` 不参与默认预处理流程。

## 5. 预期输出文件

处理后的数据建议放在 `data/processed/<category>/` 下。

每个类别预期输出：

```text
data/processed/<category>/train.tsv
data/processed/<category>/dev.tsv
data/processed/<category>/test.tsv
data/processed/<category>/seqrec_sequence.txt
data/processed/<category>/user2id.json
data/processed/<category>/id2user.json
data/processed/<category>/item2id.json
data/processed/<category>/id2item.json
data/processed/<category>/stats.json
```

其中：

- `train.tsv`：样本级训练数据，对应每个用户的前 `N-2` 个交互
- `dev.tsv`：样本级验证数据，对应每个用户的第 `N-1` 个交互
- `test.tsv`：样本级测试数据，对应每个用户的第 `N` 个交互
- `seqrec_sequence.txt`：经典 SASRec 序列格式文件
- `user2id.json`：原始用户 ID 到连续整数 ID 的映射
- `id2user.json`：连续整数 ID 到原始用户 ID 的映射
- `item2id.json`：`parent_asin` 到连续整数 ID 的映射
- `id2item.json`：连续整数 ID 到原始 `parent_asin` 的映射
- `stats.json`：该类别的数据统计摘要

## 6. SASRec 数据格式设计

后续模型同学会使用用户历史序列预测下一件商品的 `parent_asin`。

样本级 `train.tsv`、`dev.tsv`、`test.tsv` 统一使用以下字段顺序：

```text
user_id_int	target_id	rating	timestamp	seq_ids	raw_user_id	raw_parent_asin
```

字段含义：

- `user_id_int`：映射后的整数用户 ID；
- `target_id`：当前要预测的商品 `parent_asin` 映射后的整数 ID；
- `rating`：原始评分；
- `timestamp`：当前交互时间；
- `seq_ids`：history 中商品映射后的整数序列，用空格分隔；
- `raw_user_id`：原始用户 ID，便于排查；
- `raw_parent_asin`：原始商品 ID，便于排查。

经典 SASRec 序列文件 `seqrec_sequence.txt` 采用整数 ID 后的序列格式：

```text
user_id_int item_id_1 item_id_2 item_id_3 ...
```

示例：

```text
1 10 25 37 42
2 7 19 81
```

含义：

- 每一行对应一个用户；
- 第一列是重新映射后的 `user_id_int`；
- 后续列是按照时间升序排列的历史 `item_id`；
- `seqrec_sequence.txt` 主要保留给经典 SASRec 读取流程，样本级训练、验证和测试以 TSV 文件为准。

## 7. ID 映射规则

- `user_id` 从 1 开始连续编号；
- `item_id` 从 1 开始连续编号；
- `0` 保留给 padding，不分配给真实用户或真实商品；
- `train.tsv`、`dev.tsv`、`test.tsv` 必须共用同一套 `user2id.json` 和 `item2id.json`；
- 商品使用 `parent_asin` 作为推荐目标和 item 映射依据；
- 同一类别内独立建立用户和商品映射；
- 不同类别之间不共享 ID 映射。


