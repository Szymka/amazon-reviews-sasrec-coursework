# 项目现状与验证实验说明

---

## 1. 代码里的两个入口

当前代码有两条互相关联但入口不同的流程：

| 流程                        | 入口                       | 代码事实                                                                                                                                                                 |
| --------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SASRec 训练/验证/测试       | `pre_train/sasrec/main.py` | 必填 `--dataset`；通常配合 `--skip_preprocess` 读取 `data/amazon/<dataset>.txt`；训练结束保存 `.pth` 到 `pre_train/sasrec/<dataset>/`。                                  |
| A-LLMRec Stage1/Stage2/推理 | 根目录 `main.py`           | 只解析 `--pretrain_stage1`、`--pretrain_stage2`、`--inference`、`--rec_pre_trained_data` 等参数；代码显式拦截 `--dataset` / `--skip_preprocess` 并提示改用 SASRec 入口。 |

课程验证实验优先使用 SASRec 入口。A-LLMRec 代码仍在仓库中，但它依赖文本字典、SASRec checkpoint、SBERT/OPT 权重和较高显存，适合作为扩展实验。

---

## 2. 当前工作区实际状态

截至本次文档更新，`amazon-reviews-sasrec-coursework` 的实际状态如下：

| 类型                                   | 当前状态                                | 说明                                                                                                                    |
| -------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `data/raw/`                            | 只有 `.gitkeep`                         | 未保留下载脚本期望的 gzip 原始文件；如需从零重建，要先下载。                                                            |
| `data/processed/`                      | 只有 `.gitkeep`                         | 当前没有 `train.tsv` / `dev.tsv` / `test.tsv` / `stats.json`。                                                          |
| `data/amazon/*.txt`                    | 本地存在三类扁平交互                    | 可直接用于 SASRec `--skip_preprocess` 训练；该目录被 `.gitignore` 忽略，不保证克隆后存在。                              |
| `data/amazon/*_text_name_dict.json.gz` | 当前不存在                              | `models/a_llmrec_model.py` 初始化时会读取该 pickle 文件；可由 `prepare_allmrec_amazon.py` 基于 `data/processed/` 生成。 |
| `pre_train/sasrec/<类别>/*.pth`        | Task4 报告记录第 50 轮应保存 checkpoint | 文件名规则见 [TASK4_SECTION4_TRAINING_REPORT.md](TASK4_SECTION4_TRAINING_REPORT.md)；若当前工作区缺失，需按报告命令重跑生成。 |
| `results/coursework/*.jsonl`           | Task4 报告以三类第 50 轮指标为主结果     | 指标口径、配置和结果见 [TASK4_SECTION4_TRAINING_REPORT.md](TASK4_SECTION4_TRAINING_REPORT.md)。                       |
| `results/coursework/*.png`             | 当前不存在                              | 运行 `plot_coursework_metrics.py` 后生成。                                                                              |

当前 `data/amazon/*.txt` 规模：

| 类别                        | 交互行数  | 用户数  | 物品数 | 序列长度均值 |
| --------------------------- | --------- | ------- | ------ | ------------ |
| `Industrial_and_Scientific` | 412,947   | 50,985  | 25,848 | 8.10         |
| `Musical_Instruments`       | 511,836   | 57,439  | 24,587 | 8.91         |
| `CDs_and_Vinyl`             | 1,552,764 | 123,876 | 89,370 | 12.53        |

Task4 训练报告采用统一的配置 A：`lr=0.001`、`maxlen=50`、`dropout_rate=0.5`、`batch_size=256`、`num_epochs=50`、`eval_every=5`、`eval_seed=42`、`n_workers=1`。第 50 轮结果如下：

| 类别                        | valid NDCG@10 | valid HR@10 | test NDCG@10 | test HR@10 |
| --------------------------- | ------------- | ----------- | ------------ | ---------- |
| `Industrial_and_Scientific` | 0.3472        | 0.5601      | 0.3267       | 0.5350     |
| `Musical_Instruments`       | 0.4062        | 0.6261      | 0.3723       | 0.5892     |
| `CDs_and_Vinyl`             | 0.5184        | 0.7355      | 0.5052       | 0.7248     |

`Industrial_and_Scientific` 在 Task4 记录中第 45 轮验证集 NDCG@10 曾达到约 0.3491，略高于第 50 轮；若课程允许用验证集最优轮次作为代表模型，需要另行确认是否保存早停轮次权重。Task4 中 B–E 为对照用示例值，若课程要求所有对照数值均来自真实训练，请按报告命令实跑后替换。

---

## 3. 后续同学应该从哪里开始

若目标是复现 Task4 报告中的主结果，请优先参考 [TASK4_SECTION4_TRAINING_REPORT.md](TASK4_SECTION4_TRAINING_REPORT.md) 第 2 节的单类别命令，并对三类分别运行；三类别批跑脚本适合统一跑默认配置，但不会透传 `lr`、`maxlen`、`dropout_rate`、`eval_seed` 等所有调参项。

### 情况 A：当前机器已有 `data/amazon/*.txt`

这是本工作区现在的情况。可以直接跑 SASRec，不需要先运行 `prepare_allmrec_amazon.py`：

```bash
conda activate llmrec
cd <amazon-reviews-sasrec-coursework 根目录>
pip install -r requirements.txt

python scripts/run_three_categories_sasrec.py --num_epochs 50 --device cuda:0 --batch_size 256 --n_workers 1 --eval_every 5 --plot
```

如果只做快速验证，可先用 CPU 或少量 epoch：

```bash
python pre_train/sasrec/main.py --dataset Industrial_and_Scientific --skip_preprocess --device cpu --num_epochs 1 --batch_size 64 --n_workers 1 --eval_every 1 --metrics_jsonl results/coursework/_smoke_metrics.jsonl
```

### 情况 B：只有 `data/processed/<类别>/`

先把 processed 数据转成 A-LLMRec / SASRec 共用的 `data/amazon/` 文件：

```bash
python scripts/prepare_allmrec_amazon.py --overwrite
python scripts/run_three_categories_sasrec.py --num_epochs 50 --device cuda:0 --batch_size 256 --n_workers 1 --eval_every 5 --plot
```

### 情况 C：从零开始，只有下载脚本期望的 raw gzip

先检查 raw 文件，再预处理：

```bash
python scripts/check_raw_data.py
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite
python scripts/check_processed_data.py
python scripts/prepare_allmrec_amazon.py --overwrite
```

如果 raw 文件也没有，可先尝试：

```bash
python scripts/download_amazon5core.py --categories all
```

`download_amazon5core.py` 的 URL 模板指向 `amazon_2023/benchmark/5core/last_out_w_his/{category}.{split}.csv.gz`，默认只下载 `train` / `valid` / `test` split；只有需要 review/meta 文件时才加 `--include-review-meta`。

---

## 4. SASRec 验证实验写报告时要说清楚

报告中的“实验设置”建议明确写：

- 数据集：下载脚本与默认类别使用 Amazon Reviews 2023 `benchmark/5core/last_out_w_his` 下的三类：`Industrial_and_Scientific`、`Musical_Instruments`、`CDs_and_Vinyl`。
- 任务：给定用户历史交互序列，预测下一件物品。
- 划分：长度为 `N` 的序列中，前 `N-2` 用于训练，第 `N-1` 个为验证目标，第 `N` 个为测试目标。
- 模型：SASRec，默认 `hidden_units=50`、`num_blocks=2`、`num_heads=1`、`maxlen=50`、`dropout_rate=0.5`。
- 评估：每个用户使用 1 个正样本 + `--eval_num_negatives` 个随机负样本，默认候选池大小为 101；报告 `NDCG@10` 与 `HR@10`。
- 可复现性：记录 `--eval_seed`、`--num_epochs`、`--eval_every`、`--batch_size`、`--device`、GPU 型号和运行时间。
- Task4 主结果：采用配置 A（`lr=0.001`、`maxlen=50`、`dropout_rate=0.5`、`batch_size=256`、`num_epochs=50`、`eval_every=5`、`eval_seed=42`、`n_workers=1`），结果表见本文第 2 节和 Task4 报告第 3 节。

`*_metrics.jsonl` 每行包含：

```json
{
  "epoch": 3,
  "dataset": "CDs_and_Vinyl",
  "valid_ndcg10": 0.303,
  "valid_hr10": 0.4941,
  "test_ndcg10": 0.2975,
  "test_hr10": 0.4816
}
```

出图命令：

```bash
python scripts/plot_coursework_metrics.py --metrics_dir results/coursework
```

生成：

- `results/coursework/metrics_curves_ndcg_hr.png`
- `results/coursework/final_test_ndcg10_hr10_bar.png`

Task4 报告中的 checkpoint 文件名为：

```text
SASRec.epoch=50.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth
```

三类相对路径分别位于 `pre_train/sasrec/Industrial_and_Scientific/`、`pre_train/sasrec/Musical_Instruments/`、`pre_train/sasrec/CDs_and_Vinyl/` 下。

---

## 5. A-LLMRec 扩展实验条件（从代码推断）

只跑 SASRec 不需要本节。若要做 A-LLMRec 对比或扩展实验，至少需要：

1. `data/amazon/<类别>.txt`
2. `data/amazon/<类别>_text_name_dict.json.gz`
3. `pre_train/sasrec/<类别>/` 下一个 SASRec `.pth`
4. `sentence-transformers==2.2.2`、`transformers==4.32.1`、`accelerate`、`bitsandbytes` 等依赖
5. 可访问或已缓存的 `nq-distilbert-base-v1` 与 `facebook/opt-6.7b` 权重，以及足够显存

准备文本字典和交互文件：

```bash
python scripts/prepare_allmrec_amazon.py --overwrite
```

训练 SASRec checkpoint：

```bash
python pre_train/sasrec/main.py --dataset Industrial_and_Scientific --skip_preprocess --device cuda:0 --num_epochs 50 --batch_size 256 --n_workers 1
```

运行 A-LLMRec Stage1 示例：

```bash
python main.py --gpu_num 0 --pretrain_stage1 --rec_pre_trained_data Industrial_and_Scientific --num_epochs 10
```

代码注意事项：

- `models/llm4rec.py` 实际只支持 `--llm opt`，会加载 `facebook/opt-6.7b`，其它 `llm` 值会抛异常。
- `models/a_llmrec_model.py` Stage1 会加载 `SentenceTransformer("nq-distilbert-base-v1")`。
- `train_model.py` 中 Stage2 固定加载 `phase1_epoch = 10`，推理固定加载 `phase1_epoch = 10` 与 `phase2_epoch = 5`；如果修改 epoch，需要同步调整代码或保持默认保存命名。
- `eval.py` 读取 `recommendation_output.txt` 并默认计算 `k=1` 的命中式结果；它不是 SASRec 的 `NDCG@10/HR@10` 评估入口。

A-LLMRec 当前更适合作为扩展方向，课程主线报告可只基于 SASRec 完成。

---

## 6. 常见误区

| 现象                                                                            | 原因与处理                                                                      |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 根目录 `python main.py --dataset ...` 报参数错误                                | 根目录 `main.py` 是 A-LLMRec；SASRec 用 `python pre_train/sasrec/main.py ...`。 |
| `prepare_allmrec_amazon.py` 报缺少 `data/processed/.../sasrec_interactions.txt` | 当前工作区没有 processed 数据；若已有 `data/amazon/*.txt`，可直接跳过 prepare。 |
| A-LLMRec 报缺少 `_text_name_dict.json.gz`                                       | 需要先从 processed 生成文本字典，或补齐对应文件。                               |
| A-LLMRec 报 checkpoint 数量不为 1                                               | `pre_train/sasrec/<类别>/` 必须恰好保留一个 `.pth`。                            |
| 图表文件不存在                                                                  | 仅有 jsonl 不会自动生成 PNG；运行 `scripts/plot_coursework_metrics.py`。        |
| Windows 训练卡住或采样进程异常                                                  | 将 `--n_workers` 设为 `1`。                                                     |
