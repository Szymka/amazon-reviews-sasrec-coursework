# Amazon 顺序推荐 · A-LLMRec（课程数据）

本仓库在 [A-LLMRec](https://github.com/CHIANGEL/A-LLMRec)（KDD 2024）代码基础上，使用 `data/processed/` 下三类 Amazon Reviews 2023 5-core 子集：

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

每类目录中的 `sasrec_interactions.txt`（`user_id item_id` 逐行）与官方 SASRec / A-LLMRec 的 `data_partition` 格式一致；`id2item.json` 用于构造占位商品标题文本。

## 环境

本仓库约定在 **conda 环境 `llmrec`** 中运行（请自行创建并安装 PyTorch / 依赖；名称必须为 `llmrec` 时可直接复制下文命令）。

```bash
conda activate llmrec
pip install -r requirements.txt
```

若习惯非交互式调用，可使用：

```bash
conda run -n llmrec python scripts/prepare_allmrec_amazon.py --overwrite
conda run -n llmrec python main.py --help
```

以下示例均可将 `python` 换成 `conda run -n llmrec python`（在已 `activate llmrec` 的终端里则直接 `python` 即可）。

Stage 2 与推理使用 `facebook/opt-6.7b`（8-bit），需要 **NVIDIA GPU + 足够显存** 及 `bitsandbytes`。若仅验证协同过滤部分，可只训练 **SASRec** 与 A-LLMRec **Stage 1**（Sentence-BERT 对齐，仍建议 GPU）。

## 准备 `data/amazon/`（A-LLMRec 默认路径）

在项目根目录执行：

```bash
python scripts/prepare_allmrec_amazon.py --overwrite
```

将生成例如 `data/amazon/Industrial_and_Scientific.txt` 与 `data/amazon/Industrial_and_Scientific_text_name_dict.json.gz`（与原论文代码相同，为 pickle 文件，扩展名沿用 `json.gz`）。

**注意：** 根目录的 `main.py` 是 **A-LLMRec**（参数为 `--pretrain_stage1`、`--rec_pre_trained_data`、`--gpu_num` 等）。若带上 `--dataset` 或 `--skip_preprocess`，脚本会直接提示退出。训练 SASRec 请使用 **`pre_train/sasrec/main.py`**（可先 `cd pre_train/sasrec` 再 `python main.py ...`，或在根目录执行 `python pre_train/sasrec/main.py ...`）。

若遇 `cached_download` / `huggingface_hub` 相关 `ImportError`，在 `llmrec` 环境中执行 `pip install -r requirements.txt`（已上限定 `huggingface_hub<0.26`），或单独执行：`pip install "huggingface_hub>=0.14,<0.26"`。

## 1）预训练 SASRec（协同过滤骨架）

在 `pre_train/sasrec` 目录下运行（**工作目录必须是该文件夹**，以便正确解析 `../../data/amazon/`）：

```bash
cd pre_train/sasrec
python main.py --device cuda:0 --dataset Industrial_and_Scientific --skip_preprocess --num_epochs 200 --batch_size 128
```

或在**仓库根目录**直接调用 SASRec 入口（`data/amazon` 与 checkpoint 路径已按脚本位置解析，可不 `cd`）：

```bash
python pre_train/sasrec/main.py --device cuda:0 --dataset Industrial_and_Scientific --skip_preprocess --num_epochs 200 --batch_size 128
```

**不要**在根目录运行根目录的 `main.py` 并带上 `--dataset` / `--skip_preprocess`：那是 **A-LLMRec**，参数不同。

常用参数：

| 参数 | 说明 |
|------|------|
| `--skip_preprocess` | 不读 `json.gz` 元数据，使用仓库根目录已生成的 `data/amazon/<dataset>.txt` |
| `--dataset` | 与 `data/amazon/<dataset>.txt` 文件名（不含 `.txt`）一致 |
| `--n_workers` | `WarpSampler` 进程数；Windows 建议 `1` |

训练结束会在当前目录下生成 `<dataset>/SASRec.epoch=....pth`。`models/recsys_model.py` 会从 `pre_train/sasrec/<dataset>/` 目录加载 **唯一** 的 `.pth` 文件，请保持该目录内只有一个权重文件。

## 2）A-LLMRec Stage 1 / 2 / 推理

回到项目根目录：

```bash
# Stage 1：对齐协同嵌入与文本嵌入
python main.py --gpu_num 0 --pretrain_stage1 --rec_pre_trained_data Industrial_and_Scientific --num_epochs 10

# Stage 2：OPT 侧训练（需要大显存）
python main.py --gpu_num 0 --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --llm opt --num_epochs 10

# 推理（默认加载 phase1 epoch=10 与 phase2 epoch=5 的权重命名；需与训练保存一致）
python main.py --gpu_num 0 --inference --rec_pre_trained_data Industrial_and_Scientific --llm opt
python eval.py
```

多卡可加 `--multi_gpu` 并配合 `CUDA_VISIBLE_DEVICES`。

## 原始数据 → `data/processed/`（可选）

若需从官方 gzip 重新生成 `processed/`，仍可使用：

```bash
python scripts/preprocess_to_seqrec.py --categories Industrial_and_Scientific Musical_Instruments CDs_and_Vinyl --overwrite
```

（需先将官方 `train/valid/test` 放入 `data/raw/<类别>/`。）

## 目录说明

| 路径 | 作用 |
|------|------|
| `main.py` / `train_model.py` | A-LLMRec 训练与推理入口 |
| `models/` | A-LLMRec、LLM、SASRec 封装 |
| `pre_train/sasrec/` | SASRec 预训练 |
| `data/processed/` | 三类已处理子集（交互与映射） |
| `data/amazon/` | 由脚本生成的 A-LLMRec 扁平输入 |
| `scripts/prepare_allmrec_amazon.py` | 生成 `data/amazon/` |

## 引用

```bibtex
@inproceedings{chiang2024allmrec,
  title={Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System},
  author={Chiang, Wei-Yao and others},
  booktitle={KDD},
  year={2024}
}
```
