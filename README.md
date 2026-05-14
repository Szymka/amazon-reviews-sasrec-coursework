# Amazon 顺序推荐（5-Core）· SASRec / A-LLMRec

三类数据：**Industrial_and_Scientific**、**Musical_Instruments**、**CDs_and_Vinyl**（Amazon Reviews 2023 **5-Core**）。

| 你想…                                             | 打开                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| **了解当前项目状态、已有数据/结果、代码入口关系** | **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**（建议先看）     |
| **按清单训练、验证、写报告、查命令**              | **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**             |
| 核对 **N−2/N−1/N**、TSV 字段、NDCG@10 与代码      | [docs/COURSEWORK_DATA_AND_EVAL.md](docs/COURSEWORK_DATA_AND_EVAL.md) |
| 文档索引                                          | [docs/README.md](docs/README.md)                                     |

---

## 环境与 conda（全体统一）

```bash
conda activate llmrec
cd <本仓库根目录>
pip install -r requirements.txt
```

未激活环境时：`conda run -n llmrec python ...`。

---

## 最短上手（当前已有 `data/amazon/*.txt` 时）

在**仓库根目录**依次执行：

```bash
python scripts/run_three_categories_sasrec.py --num_epochs 50 --device cuda:0 --batch_size 256 --n_workers 1 --eval_every 5 --plot
```

当前工作区已有本地 `data/amazon/Industrial_and_Scientific.txt`、`Musical_Instruments.txt`、`CDs_and_Vinyl.txt`，可直接用 `--skip_preprocess` 流程训练。产出见 **`results/coursework/`**（`*_metrics.jsonl` + PNG）。`data/amazon/` 与上述结果**默认不纳入 Git**（见 `.gitignore`）；克隆后若缺这些本地数据，请按 **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** 的“从零开始”流程重建。

若只有 `data/processed/<类别>/`，先生成 `data/amazon/`：

```bash
python scripts/prepare_allmrec_amazon.py --overwrite
```

若尚无 `data/processed/`，需先有 `data/raw/<类别>/*.csv.gz`，再：

```bash
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite
```

更细步骤、当前项目状态、报告要写什么、参数表见 **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** 与 **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**。

---

## 训练 vs 验证：入口不要混

| 做什么                         | 命令入口                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| **SASRec** 训练与验证/测试评估 | `python pre_train/sasrec/main.py ...`（支持 `--dataset`、`--skip_preprocess`、`--batch_size`） |
| **A-LLMRec**                   | 根目录 `python main.py --pretrain_stage1 --rec_pre_trained_data ...`（**不要**带 `--dataset`） |

根目录误带 `--dataset` / `--skip_preprocess` 时会提示改用 SASRec 脚本。

### A-LLMRec（可选扩展）

需先在 `pre_train/sasrec/<类别>/` 保留**唯一** `.pth`。根目录示例：

```bash
python main.py --gpu_num 0 --pretrain_stage1 --rec_pre_trained_data Industrial_and_Scientific --num_epochs 10
```

Stage2 / 推理依赖 `models/a_llmrec_model.py`、`models/llm4rec.py` 中加载的 SBERT/OPT 权重与显存；具体条件见 `docs/PROJECT_STATUS.md`。

---

## 脚本与目录速查

| 路径                                     | 作用                                                            |
| ---------------------------------------- | --------------------------------------------------------------- |
| `docs/PROJECT_STATUS.md`                 | 当前项目状态、代码入口关系、已有数据/结果、复现实验入口         |
| `docs/EXPERIMENT_GUIDE.md`               | 实验清单、速查命令、报告要点、`metrics` 字段                    |
| `docs/COURSEWORK_DATA_AND_EVAL.md`       | 数据与指标定义                                                  |
| `scripts/preprocess_to_seqrec.py`        | raw → `data/processed/`                                         |
| `scripts/prepare_allmrec_amazon.py`      | → `data/amazon/`                                                |
| `scripts/run_three_categories_sasrec.py` | 三类别 SASRec + 可选 `--plot`                                   |
| `scripts/plot_coursework_metrics.py`     | 从 jsonl 出图                                                   |
| `pre_train/sasrec/main.py`               | 单类 SASRec                                                     |
| `results/coursework/`                    | 指标与图（**默认不提交**，见 `.gitignore`；报告里可贴图或附表） |

**Git 策略**：`data/raw/`、`data/processed/`、`data/amazon/`、`**/*.pth`、`**/*.pt`、`results/**` 等均为忽略项，仓库只保留脚本与文档；克隆后请按实际已有数据状态重建或拷贝本地数据，再训练生成指标。

---

## 依赖与排错（摘要）

- `huggingface_hub<0.26`：兼容 A-LLMRec 的 `sentence-transformers==2.2.2`；仅跑 SASRec 若报错也可 `pip install -r requirements.txt`。
- `transformers` 的 `FutureWarning` 可忽略。

更多排错表：**[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md#常见问题排错)**。

---

## 说明

本文档中的流程说明按本仓库代码校准；如旧实验记录或外部说明与代码不一致，以 `docs/PROJECT_STATUS.md` 和实际脚本参数为准。
