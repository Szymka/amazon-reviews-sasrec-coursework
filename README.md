# Amazon 顺序推荐（5-Core）· SASRec / A-LLMRec

本仓库面向 **Amazon Reviews 2023 5-Core** 下三个类别：**Industrial_and_Scientific**、**Musical_Instruments**、**CDs_and_Vinyl**。数据划分、TSV 字段与 **NDCG@10 / HR@10** 评估方式见 **[docs/COURSEWORK_DATA_AND_EVAL.md](docs/COURSEWORK_DATA_AND_EVAL.md)**（建议先读）。

---

## 环境与 conda

在 **`llmrec`** conda 环境中操作：

```bash
conda activate llmrec
cd <本仓库根目录>
pip install -r requirements.txt
```

非交互式：

```bash
conda run -n llmrec pip install -r requirements.txt
```

---

## 数据划分（摘要）

对每个用户，按时间的完整交互序列长度为 **N**：

| 子集 | 内容 |
|------|------|
| 训练 | 前 **N−2** 个交互 |
| 验证 | 第 **N−1** 个交互（`dev.tsv` / 官方 valid） |
| 测试 | 第 **N** 个交互（`test.tsv` / 官方 test） |

`train.tsv` / `dev.tsv` / `test.tsv` 含 **`user_id_int`（用户）**、**`target_id` / `raw_parent_asin`（商品）**、**`rating`、`timestamp`**、**`seq_ids`（history 对应的整数序列）**、**`raw_user_id`** 等，语义为：用户在对 **history** 中最后一个商品交互之后，又对当前行的 **parent_asin** 产生交互；验证/测试即根据 history **预测该 parent_asin**。

由 `scripts/preprocess_to_seqrec.py` 从 `data/raw/<类别>/` 生成 `data/processed/<类别>/`；**5-Core** 保证用户与商品交互次数下限，减轻稀疏。

---

## 推荐流程（训练 → 验证 / 测试）

以下均在**仓库根目录**执行（除非注明）。

### 0）准备 processed 与 `data/amazon/`（首次或更新数据后）

```bash
# 若尚无 processed，需先放入官方 gzip 再执行：
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite

# 生成 SASRec / A-LLMRec 共用的 user-item 行文件：
python scripts/prepare_allmrec_amazon.py --overwrite
```

### 1）训练（SASRec）

**入口必须是** `pre_train/sasrec/main.py`（**不要**用根目录 `main.py` 跑 `--dataset`，那是 A-LLMRec）。

单类别示例（训练集上 BCE；按 `--eval_every` 在**验证集 + 测试集**上算 **NDCG@10、HR@10**，候选为 1 正 + 100 负，可 `--eval_num_negatives` 调整）：

```bash
python pre_train/sasrec/main.py ^
  --dataset Industrial_and_Scientific ^
  --skip_preprocess ^
  --device cuda:0 ^
  --num_epochs 200 ^
  --batch_size 128 ^
  --n_workers 1 ^
  --eval_every 20 ^
  --metrics_jsonl results/coursework/Industrial_and_Scientific_metrics.jsonl
```

（Linux/macOS 将 `^` 换为行末 `\`。）

**三类别顺序训练并写指标、可选自动画图：**

```bash
python scripts/run_three_categories_sasrec.py --num_epochs 5 --device cuda:0 --batch_size 256 --n_workers 1 --eval_every 1 --plot
```

仅根据已有 `*_metrics.jsonl` 出图：

```bash
python scripts/plot_coursework_metrics.py --metrics_dir results/coursework
```

默认输出：`results/coursework/metrics_curves_ndcg_hr.png`（各 epoch 曲线）、`final_test_ndcg10_hr10_bar.png`（最后一轮测试集对比）。

### 2）验证 / 测试在代码中的对应关系

- **验证集指标**：用 **训练前缀** 构造序列，预测 **第 N−1 个物品**（与 `dev.tsv` 目标一致）。
- **测试集指标**：用 **训练前缀 + 验证物品** 构造序列，预测 **第 N 个物品**（与 `test.tsv` 目标一致）。
- 指标含义见 `docs/COURSEWORK_DATA_AND_EVAL.md`（随机负采样、用户子采样规则等）。

### 3）A-LLMRec（可选）

根目录 `main.py` 为 **A-LLMRec**（`--pretrain_stage1`、`--rec_pre_trained_data`、`--gpu_num` 等）。需先在 `pre_train/sasrec/<类别>/` 下保留**唯一** SASRec `.pth`。详见原论文流程；数据划分与上表一致。

若根目录命令误含 `--dataset` / `--skip_preprocess`，脚本会提示改用 SASRec 入口。

---

## 常用路径

| 路径 | 作用 |
|------|------|
| `docs/COURSEWORK_DATA_AND_EVAL.md` | 划分、字段、NDCG@10、操作顺序 |
| `scripts/preprocess_to_seqrec.py` | raw gzip → `data/processed/` |
| `scripts/prepare_allmrec_amazon.py` | → `data/amazon/` |
| `scripts/run_three_categories_sasrec.py` | 三类别 SASRec + metrics |
| `scripts/plot_coursework_metrics.py` | 评估可视化 |
| `pre_train/sasrec/main.py` | SASRec 训练与评估 |
| `data/processed/<类别>/` | TSV、映射、`sasrec_interactions.txt` |
| `data/amazon/` | 扁平交互，供 SASRec `--skip_preprocess` |
| `results/coursework/` | `*_metrics.jsonl` 与 PNG |

---

## 依赖与排错

- `requirements.txt` 含 `huggingface_hub<0.26` 以兼容 `sentence-transformers==2.2.2`（A-LLMRec Stage1）。若遇 `cached_download` 报错：`pip install -r requirements.txt`。
- `transformers` 与 PyTorch 的 `FutureWarning` 可忽略。

---

## 引用（A-LLMRec）

```bibtex
@inproceedings{chiang2024allmrec,
  title={Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System},
  author={Chiang, Wei-Yao and others},
  booktitle={KDD},
  year={2024}
}
```
