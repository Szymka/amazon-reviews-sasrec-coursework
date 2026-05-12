# Amazon 顺序推荐（5-Core）· SASRec / A-LLMRec

三类数据：**Industrial_and_Scientific**、**Musical_Instruments**、**CDs_and_Vinyl**（Amazon Reviews 2023 **5-Core**）。

| 你想… | 打开 |
|--------|------|
| **按清单训练、验证、写报告、查命令** | **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**（推荐先看） |
| 核对 **N−2/N−1/N**、TSV 字段、NDCG@10 与代码 | [docs/COURSEWORK_DATA_AND_EVAL.md](docs/COURSEWORK_DATA_AND_EVAL.md) |
| 文档索引 | [docs/README.md](docs/README.md) |

---

## 环境与 conda（全体统一）

```bash
conda activate llmrec
cd <本仓库根目录>
pip install -r requirements.txt
```

未激活环境时：`conda run -n llmrec python ...`。

---

## 最短上手（已有 `data/processed/` 时）

在**仓库根目录**依次执行：

```bash
python scripts/prepare_allmrec_amazon.py --overwrite
python scripts/run_three_categories_sasrec.py --num_epochs 50 --device cuda:0 --batch_size 256 --n_workers 1 --eval_every 5 --plot
```

产出见 **`results/coursework/`**（`*_metrics.jsonl` + PNG）。`data/amazon/` 与上述结果**默认不纳入 Git**（见 `.gitignore`）。更细步骤、报告要写什么、参数表见 **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**。

若尚无 `data/processed/`，需先有 `data/raw/<类别>/*.csv.gz`，再：

```bash
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite
```

---

## 训练 vs 验证：入口不要混

| 做什么 | 命令入口 |
|--------|----------|
| **SASRec** 训练与验证/测试评估 | `python pre_train/sasrec/main.py ...`（支持 `--dataset`、`--skip_preprocess`、`--batch_size`） |
| **A-LLMRec** | 根目录 `python main.py --pretrain_stage1 --rec_pre_trained_data ...`（**不要**带 `--dataset`） |

根目录误带 `--dataset` / `--skip_preprocess` 时会提示改用 SASRec 脚本。

### A-LLMRec（可选扩展）

需先在 `pre_train/sasrec/<类别>/` 保留**唯一** `.pth`。根目录示例：

```bash
python main.py --gpu_num 0 --pretrain_stage1 --rec_pre_trained_data Industrial_and_Scientific --num_epochs 10
```

Stage2 / 推理依赖大模型与显存，详见原论文与历史 README 说明；数据划分与 `COURSEWORK` 文档一致。

---

## 脚本与目录速查

| 路径 | 作用 |
|------|------|
| `docs/EXPERIMENT_GUIDE.md` | 实验清单、速查命令、报告要点、`metrics` 字段 |
| `docs/COURSEWORK_DATA_AND_EVAL.md` | 数据与指标定义 |
| `scripts/preprocess_to_seqrec.py` | raw → `data/processed/` |
| `scripts/prepare_allmrec_amazon.py` | → `data/amazon/` |
| `scripts/run_three_categories_sasrec.py` | 三类别 SASRec + 可选 `--plot` |
| `scripts/plot_coursework_metrics.py` | 从 jsonl 出图 |
| `pre_train/sasrec/main.py` | 单类 SASRec |
| `results/coursework/` | 指标与图（**默认不提交**，见 `.gitignore`；报告里可贴图或附表） |

**Git 策略**：`data/raw/`、`data/processed/`、`data/amazon/`、`**/*.pth`、`**/*.pt`、`results/**` 等均为忽略项，仓库只保留脚本与文档；克隆后请本地跑 `preprocess` / `prepare_allmrec_amazon` / 训练生成数据与指标。

---

## 依赖与排错（摘要）

- `huggingface_hub<0.26`：兼容 A-LLMRec 的 `sentence-transformers==2.2.2`；仅跑 SASRec 若报错也可 `pip install -r requirements.txt`。
- `transformers` 的 `FutureWarning` 可忽略。

更多排错表：**[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md#常见问题排错)**。

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
