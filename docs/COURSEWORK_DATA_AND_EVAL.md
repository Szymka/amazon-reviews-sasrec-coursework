# 课程数据划分、训练与验证说明

本文档与 `scripts/preprocess_to_seqrec.py`、`pre_train/sasrec/` 中的 SASRec 实现一致，便于对照作业要求。

## 数据来源与 5-Core

- 使用 **Amazon Reviews 官方 5-Core** 子集：每位用户、每个商品在子集中至少出现 **5** 次交互，缓解极端稀疏。
- 工程内三类：**Industrial_and_Scientific**、**Musical_Instruments**、**CDs_and_Vinyl**。
- 原始 gzip 分片放在 `data/raw/<类别>/`（`train` / `valid` / `test`），由 `scripts/preprocess_to_seqrec.py` 生成 `data/processed/<类别>/`。

## 用户序列与划分（长度 N）

对每个用户，将其**按时间排序**的完整交互序列记为长度 \(N\) 的序列 \(i_1,\ldots,i_N\)（商品已映射为整数 `target_id` / `item_id`）。

| 子集 | 含义 | 在 SASRec 中的用法 |
|------|------|---------------------|
| **训练** | 前 **N−2** 个交互 | `user_train[u]`，用于 BCE 训练与负采样 |
| **验证** | 第 **N−1** 个交互 | `user_valid[u]`，评估时用 `train` 为历史、预测该物品 |
| **测试** | 第 **N** 个交互 | `user_test[u]`，评估时用 `train ∪ {验证物品}` 为历史、预测该物品 |

交互数 **&lt; 3** 的用户：全部划入训练，不参与 valid/test 评估（与 SASRec 参考实现一致）。

## TSV 字段（`train.tsv` / `dev.tsv` / `test.tsv`）

与官方导出及作业描述对齐的语义如下（列名以仓库实际文件为准）：

| 列名 | 含义 |
|------|------|
| `user_id_int` | 用户整数 ID（从 1 连续编号） |
| `target_id` | 当前行要预测的商品整数 ID（对应 `parent_asin` 映射） |
| `rating` | 评分 |
| `timestamp` | 时间戳 |
| `seq_ids` | **history**：该行发生前用户已交互的商品 ID 序列（空格分隔），即「评论完 history 最后一个商品后再评论 target」 |
| `raw_user_id` | 原始用户 ID（可关联 User Review） |
| `raw_parent_asin` | 原始商品 ID（可关联 Item Metadata） |

- **验证 / 测试文件**中每一行：在给定 `history`（及 `seq_ids`）条件下，预测 `target_id` / `raw_parent_asin`。
- `dev.tsv` 对应官方 **valid** 分片；`test.tsv` 对应 **test** 分片。

## `sasrec_interactions.txt` 与 `data/amazon/*.txt`

- `sasrec_interactions.txt`：按用户、按时间展开为「`user_id item_id`」多行，与 SASRec `data_partition` 读入格式一致。
- `python scripts/prepare_allmrec_amazon.py` 将其复制为 `data/amazon/<类别>.txt`，供 `pre_train/sasrec/main.py --skip_preprocess` 使用。

## 评估指标：NDCG@10 与 HR@10

在 `pre_train/sasrec/utils.py` 中：

- **验证**：输入序列为 **训练前缀**；候选为 **1 个真实验证物品 + `eval_num_negatives` 个随机负样本**（默认 100，共 101 候选）。若真实物品排名位置 `rank < 10`（0-based 升序排名中的前 10），则计为命中 **HR@10**；并累加 **NDCG@10** 常用增益 \(1/\log_2(\mathrm{rank}+2)\)。
- **测试**：输入序列为 **训练前缀 + 验证目标**；候选为 **1 个真实测试物品 + 同样数量的负样本**；同样按 **NDCG@10 / HR@10** 统计。负采样会排除已出现在 train / valid / test 目标中的物品，避免 trivial 重复。

用户量 &gt; 10000 时，评估对用户随机子采样 10000（与参考实现一致）。

## 推荐操作顺序（conda 环境 `llmrec`）

在仓库根目录：

1. **安装依赖**  
   `conda activate llmrec`  
   `pip install -r requirements.txt`

2. **（可选）从 raw 重新生成 processed**  
   `python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite`

3. **生成 SASRec / A-LLMRec 共用的扁平交互**  
   `python scripts/prepare_allmrec_amazon.py --overwrite`

4. **单类别训练 + 记录指标**  
   `python pre_train/sasrec/main.py --dataset Industrial_and_Scientific --skip_preprocess --device cuda:0 --num_epochs 200 --batch_size 128 --n_workers 1 --metrics_jsonl results/coursework/Industrial_and_Scientific_metrics.jsonl --eval_every 20`

5. **三类别批跑 + 画图**  
   `python scripts/run_three_categories_sasrec.py --num_epochs 5 --device cuda:0 --plot`

6. **仅根据已有 jsonl 出图**  
   `python scripts/plot_coursework_metrics.py --metrics_dir results/coursework`

图表默认输出：`results/coursework/metrics_curves_ndcg_hr.png`（按 epoch 曲线）、`final_test_ndcg10_hr10_bar.png`（最后一轮测试集柱状对比）。

## A-LLMRec（根目录 `main.py`）

在 SASRec 训练完成且 `pre_train/sasrec/<类别>/` 下仅保留一个 `.pth` 后，可在根目录按 README 进行 Stage1 / Stage2；其数据划分与上文相同（同样读取 `data/amazon/<类别>.txt`）。

---

## 延伸阅读（给写报告、做实验的同学）

- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**：实验前检查清单、一键命令、参数表、`metrics` 各字段、报告章节提示、常见问题。  
- **[README.md](../README.md)**：仓库入口、最短上手、脚本索引。

写「实验设置」时可从本页摘录：**5-Core**、**N−2/N−1/N**、**NDCG@10 / HR@10**（101 候选、前 10 命中）、**eval 子采样 10000 用户**（若适用）。写「复现」时请记录：`--eval_seed`、`--eval_num_negatives`、`--eval_every`、`--num_epochs`、硬件与 `llmrec` 中 `torch` 版本。
