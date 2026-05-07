# Team Handoff

本文档是当前数据预处理工作的交接说明，供模型训练、评估和报告撰写同学继续使用。

## 已完成工作

已完成以下部分：

- 下载 Amazon Reviews 2023 5-Core 三个类别数据。
- 检查三个类别 raw 数据目录、15 个 raw 文件、CSV 字段和 JSONL 可读性。
- 编写并完善 `scripts/check_raw_data.py`。
- 编写并完善 `scripts/preprocess_to_sasrec.py`。
- 将三个类别转换为 SASRec 可用的 processed 数据。
- 编写并完善 `scripts/check_processed_data.py`。
- 三个类别 processed 输出均已通过检查。

## 已处理类别

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

## processed 输出文件

每个类别目录 `data/processed/<category>/` 下都有：

```text
train.tsv
dev.tsv
test.tsv
sasrec_sequence.txt
sasrec_interactions.txt
user2id.json
id2user.json
item2id.json
id2item.json
stats.json
```

## 核心统计

| category | users | items | train_rows | dev_rows | test_rows | sasrec_sequences | sasrec_interactions | min_seq | max_seq | avg_seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Industrial_and_Scientific | 50985 | 25848 | 310977 | 50985 | 50985 | 50985 | 412947 | 5 | 204 | 8.099382 |
| Musical_Instruments | 57439 | 24587 | 396958 | 57439 | 57439 | 57439 | 511836 | 5 | 288 | 8.910949 |
| CDs_and_Vinyl | 123876 | 89370 | 1305012 | 123876 | 123876 | 123876 | 1552764 | 5 | 1912 | 12.534825 |

三个类别的 `dev.tsv` 均来自 raw 阶段的 `<category>.valid.csv.gz`。

## 后续接手方式

模型训练同学：

- 从 `data/processed/<category>/` 读取数据。
- 优先使用 `sasrec_sequence.txt` 或 `train.tsv/dev.tsv/test.tsv`。
- 使用 `configs/` 中的 YAML 作为起始配置。
- 不要写死本机绝对路径。
- 不要把 checkpoint 提交到 GitHub。

评估同学：

- 从 `test.tsv` 和模型输出计算 `HitRate@10`、`NDCG@10`、`Recall@10`。
- 将最终表格放入 `results/tables/`。
- 将图表放入 `results/figures/`。
- 三个类别分别汇报结果。

报告同学：

- 数据来源、划分逻辑和输出格式可引用 `docs/DATA_PREPROCESS.md`。
- 类别规模和序列长度可引用各类别的 `stats.json` 或上方统计表。
- 报告中明确 raw 阶段 `valid.csv.gz` 对应 processed 阶段 `dev.tsv`。

## GitHub 注意事项

不要把以下目录中的完整数据上传到 GitHub：

```text
data/raw/
data/processed/
```

这些目录只在本地保留完整数据；仓库中只提交 `.gitkeep` 占位文件。
