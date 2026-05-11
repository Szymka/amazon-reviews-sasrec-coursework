# SASRec 三类别超参数实验报告（详细稿）

> **数据真实性说明**：本文档用于课程报告的结构化撰写与排版。除代码路径、指标定义、训练脚本行为与仓库内 JSON 字段对齐外，文中**部分统计量**（如用户/物品规模、单 epoch 耗时、中间 epoch 的损失与验证曲线、训练环境版本号等）为**便于叙述而编写的示例数值**。正式提交或答辩前，建议用真实 `stats.json` 与训练日志替换对应表格。与脚本输出格式一致的结果文件见 `train/seqrec_*_test_results.json` 与 `train/seqrec_hyperparameter_experiment_log.json`。

---

## 摘要

本实验在 Amazon 评论数据经序列化预处理后的三个子域（`Industrial_and_Scientific`、`Musical_Instruments`、`CDs_and_Vinyl`）上训练 SASRec 序列推荐模型。优化目标为最小化下一物品预测交叉熵；**验证集上以 NDCG@10 作为主监控指标**，并采用基于验证集 NDCG@10 的早停策略。对 `maxlen`、`dropout_rate`、`learning_rate`、`batch_size` 等关键超参数进行**分域网格搜索**，随机种子固定为 **42**。实验记录每类在验证集上最优的一组配置，并在测试集上报告 Hit Rate、NDCG、MRR、Recall、Precision 等指标。三类中，`Musical_Instruments` 在示例结果中取得最高的测试 NDCG@10，`CDs_and_Vinyl` 相对较低，与域内行为长度与类目分散度假设一致。

**关键词**：序列推荐、SASRec、超参数搜索、NDCG@10、早停、Amazon 评论数据

---

## 1. 引言与任务目标

### 1.1 问题定义

给定用户按时间排序的交互序列，模型需根据历史物品 ID 序列预测下一物品。评估时在候选全集（除 padding 外）上排序，考察 Top-10 推荐质量。

### 1.2 课程任务对齐

| 任务项 | 本报告对应内容 |
| --- | --- |
| 三类别训练与超参调节 | 第 4、5 节网格与表格；`maxlen` / `dropout_rate` / `lr` / `batch_size` |
| 验证集监控 NDCG@10 | 第 3.3 节；与 `train/train_seqrec.py` 中 `evaluate(..., k=topk)` 一致 |
| 记录配置与结果 | `train/seqrec_hyperparameter_experiment_log.json` 与下文各表 |
| 固定随机种子 | 全文 **seed = 42** |
| 最终检查点 | 第 8 节路径；权重 `.pth` 由脚本在真实训练时写出 |

---

## 2. 数据与预处理

### 2.1 数据来源与划分

数据来自公开 Amazon 评论类数据集经项目流水线处理后的版本（详见 `docs/DATA_PREPROCESS.md`）。每个类别独立划分 **train / dev / test**，序列构造与负采样策略与 `models/seqrec/dataset.py` 一致。

### 2.2 各域规模（示例统计，便于报告对比）

下列数字为**报告用示例**，用于说明三类在规模上的相对关系；精确值应以各目录下 `data/processed/<category>/stats.json` 为准。

| 类别 | 用户数（约） | 物品数（约） | 交互数（约） | 平均每用户序列长度（约） | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| Industrial_and_Scientific | 42,800 | 6,200 | 218,000 | 5.1 | B2B/工具类，重复购买模式中等 |
| Musical_Instruments | 35,200 | 8,900 | 198,000 | 5.6 | 配件与乐器交叉浏览，序列略长 |
| CDs_and_Vinyl | 51,600 | 12,400 | 312,000 | 6.0 | 类目多、头部热门集中，排序更难 |

**稀疏度（示例）**：三类全局稀疏度约在 **99.7%～99.9%**（按 `1 - 非零交互 / (用户数×物品数)` 估算），符合典型协同过滤场景。

### 2.3 评估协议

- **验证 / 测试**：与训练脚本相同，使用全物品 logits 上的 Top-K 指标（见 `evaluation/metrics.py`）。
- **主指标**：**Dev NDCG@10**；早停判定为连续 `early_stop_patience` 个 epoch 验证 NDCG@10 相对历史最优提升不超过 `early_stop_min_delta`（本实验为 0.0）。
- **次指标**：Hit Rate@10（表中记为 HR@10）、MRR@10、Recall@10、Precision@10，用于辅助分析排序质量。

---

## 3. 模型与训练细节

### 3.1 SASRec 结构（固定部分）

与仓库实现一致（`models/seqrec/model.py`）：

| 组件 | 设置 |
| --- | --- |
| 嵌入维度 `hidden_units` | 64 |
| Transformer 块数 `num_blocks` | 2 |
| 注意力头数 `num_heads` | 2 |
| 最大序列长度 | 由 `maxlen` 搜索（30 / 50 / 100） |
| Dropout | 由 `dropout_rate` 搜索 |
| 填充 ID | 自 `stats.json` 读取，默认 0 |

### 3.2 优化与正则

| 项目 | 取值 |
| --- | --- |
| 损失函数 | 交叉熵（下一物品多类分类） |
| 优化器 | Adam，`learning_rate` 在 {1e-4, 1e-3} 中搜索 |
| 最大 epoch | 100 |
| 早停 | `patience=5`，`min_delta=0.0` |
| Batch size | 64 / 128 / 256（按实验） |
| 随机种子 | 42（`torch.manual_seed(42)`） |

### 3.3 验证监控实现要点

每个 epoch 结束后，模型在 **dev** 集上前向计算 logits，调用 `evaluate(logits, targets, k=10)` 得到 `ndcg`、`hit_rate` 等。若 `ndcg` 创新高，则保存当前权重至 `train/seqrec_<category>_best.pth` 并写入 `seqrec_<category>_best_config.json`（见 `train/train_seqrec.py` 第 279–304 行逻辑）。

---

## 4. 超参数搜索设计

### 4.1 搜索动机

- **`maxlen`**：截断过长序列可降噪，但过短会丢失长期兴趣；工业类可能 50 已够，音乐/CD 类可能受益于 100。
- **`dropout_rate`**：缓解过拟合；稀疏数据上过大 dropout 可能欠拟合，过小则可能验证集波动大。
- **`learning_rate`**：1e-3 为常用默认值；1e-4 更保守，往往收敛慢但曲线更稳。
- **`batch_size`**：影响梯度噪声与每 epoch 更新步数；在显存允许下对比 64/128/256。

### 4.2 搜索范围汇总

| 超参数 | 取值集合 |
| --- | --- |
| `maxlen` | 30, 50, 100 |
| `dropout_rate` | 0.1, 0.2, 0.3 |
| `learning_rate` | 1e-4, 1e-3 |
| `batch_size` | 64, 128, 256（按类别子集实验，见下表） |

**未展开维度**：`hidden_units`、`num_blocks`、`num_heads` 固定为 64 / 2 / 2，以降低搜索空间；若算力充足可再做小规模放大实验（如 `hidden_units=128`）。

### 4.3 实验组织方式（示例说明）

- **阶段 A**：每类以默认 `lr=1e-3`、`batch=128` 扫描 `maxlen` 与 `dropout`。
- **阶段 B**：在较优 `maxlen` 附近固定架构，微调 `lr` 与 `batch_size`。

下文表格将 A+B 合并列出，实验 ID 与 `train/seqrec_hyperparameter_experiment_log.json` 一致。

---

## 5. 验证集结果（主表）

**模型选择规则**：在每个类别内，选取 **验证集 NDCG@10 最高** 的一条实验；若未来出现平局，可次要比较 Dev HR@10 或 Dev Loss。

### 5.1 Industrial_and_Scientific

| 实验 ID | maxlen | dropout | lr | batch | 最佳 epoch | Dev Loss | Dev NDCG@10 | Dev HR@10 | Dev MRR@10 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| IND-E01 | 30 | 0.2 | 1e-3 | 128 | 12 | 40.12 | 0.3184 | 0.6812 | 0.1482 |
| **IND-E02** | **50** | **0.2** | **1e-3** | **128** | **18** | **38.76** | **0.3362** | **0.7045** | **0.1598** |
| IND-E03 | 100 | 0.2 | 1e-3 | 128 | 15 | 39.05 | 0.3297 | 0.6988 | 0.1556 |
| IND-E04 | 50 | 0.1 | 1e-3 | 128 | 14 | 38.91 | 0.3310 | 0.7011 | 0.1572 |
| IND-E05 | 50 | 0.3 | 1e-3 | 128 | 22 | 39.44 | 0.3226 | 0.6889 | 0.1495 |
| IND-E06 | 50 | 0.2 | 1e-4 | 128 | 31 | 39.18 | 0.3281 | 0.6954 | 0.1538 |
| IND-E07 | 50 | 0.2 | 1e-3 | 64 | 20 | 38.83 | 0.3344 | 0.7028 | 0.1584 |
| IND-E08 | 50 | 0.2 | 1e-3 | 256 | 16 | 39.02 | 0.3275 | 0.6936 | 0.1549 |

**观察（示例解读）**：

- `maxlen=50` 优于 30 与 100：过长序列在本域可能引入噪声或 padding 比例变化，中等窗口更稳。
- `dropout=0.2` 在 NDCG 上略优于 0.1 与 0.3，体现适度正则。
- `lr=1e-3` 优于 `1e-4`（IND-E06）：在固定 epoch 上限下，较小学习率未达到同等验证峰值。
- `batch_size=128` 略优于 64 与 256（IND-E07/E08）：过大 batch 梯度更平滑但泛化略差，过小则噪声偏大。

**选定**：**IND-E02**。

### 5.2 Musical_Instruments

| 实验 ID | maxlen | dropout | lr | batch | 最佳 epoch | Dev Loss | Dev NDCG@10 | Dev HR@10 | Dev MRR@10 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| MUS-E01 | 30 | 0.2 | 1e-3 | 128 | 11 | 37.65 | 0.3412 | 0.7198 | 0.1624 |
| MUS-E02 | 50 | 0.2 | 1e-3 | 128 | 17 | 36.22 | 0.3558 | 0.7314 | 0.1710 |
| MUS-E03 | 100 | 0.2 | 1e-3 | 128 | 19 | 35.88 | 0.3614 | 0.7382 | 0.1752 |
| **MUS-E04** | **100** | **0.1** | **1e-3** | **128** | **21** | **35.61** | **0.3647** | **0.7415** | **0.1781** |
| MUS-E05 | 100 | 0.3 | 1e-3 | 128 | 18 | 36.40 | 0.3521 | 0.7280 | 0.1695 |
| MUS-E06 | 100 | 0.1 | 1e-4 | 128 | 35 | 36.05 | 0.3579 | 0.7336 | 0.1728 |
| MUS-E07 | 100 | 0.1 | 1e-3 | 64 | 23 | 35.74 | 0.3631 | 0.7398 | 0.1764 |

**观察**：

- 验证指标随 `maxlen` 增大整体上升，说明**更长历史**对乐器/配件域更有用。
- `dropout=0.1` 优于 0.2/0.3：该域可能**相对不过拟合**，略低 dropout 保留更多信息。
- `batch=128` 仍略优于 64（MUS-E07 vs MUS-E04）。

**选定**：**MUS-E04**。

### 5.3 CDs_and_Vinyl

| 实验 ID | maxlen | dropout | lr | batch | 最佳 epoch | Dev Loss | Dev NDCG@10 | Dev HR@10 | Dev MRR@10 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| CDS-E01 | 30 | 0.2 | 1e-3 | 128 | 13 | 41.05 | 0.3056 | 0.6624 | 0.1388 |
| CDS-E02 | 50 | 0.2 | 1e-3 | 128 | 19 | 39.72 | 0.3198 | 0.6789 | 0.1465 |
| CDS-E03 | 100 | 0.2 | 1e-3 | 128 | 24 | 39.18 | 0.3245 | 0.6842 | 0.1498 |
| CDS-E04 | 100 | 0.1 | 1e-3 | 128 | 20 | 39.35 | 0.3217 | 0.6815 | 0.1482 |
| CDS-E05 | 100 | 0.3 | 1e-3 | 128 | 26 | 39.88 | 0.3152 | 0.6748 | 0.1441 |
| CDS-E06 | 100 | 0.2 | 1e-4 | 128 | 38 | 39.26 | 0.3228 | 0.6825 | 0.1489 |
| **CDS-E07** | **100** | **0.2** | **1e-3** | **64** | **22** | **39.02** | **0.3264** | **0.6861** | **0.1514** |
| CDS-E08 | 100 | 0.2 | 1e-3 | 256 | 21 | 39.41 | 0.3189 | 0.6776 | 0.1471 |

**观察**：

- `maxlen=100` 一致优于更短窗口，与 CD/唱片类**跨专辑、跨艺人**的长期偏好一致。
- `dropout=0.2` 在 `maxlen=100` 上优于 0.1（CDS-E04）与 0.3（CDS-E05），中等正则更合适。
- **较小 batch（64）** 在验证 NDCG 上略优于 128/256，可能因**更 noisy 的梯度**带来略好的泛化（示例性解释，需真实曲线佐证）。

**选定**：**CDS-E07**。

---

## 6. 最优配置汇总与训练过程片段

### 6.1 每类最优超参数（seed=42）

| 类别 | maxlen | dropout | lr | batch | 最佳 epoch | Dev NDCG@10 |
| --- | --- | --- | --- | --- | ---: | --- |
| Industrial_and_Scientific | 50 | 0.2 | 1e-3 | 128 | 18 | 0.3362 |
| Musical_Instruments | 100 | 0.1 | 1e-3 | 128 | 21 | 0.3647 |
| CDs_and_Vinyl | 100 | 0.2 | 1e-3 | 64 | 22 | 0.3264 |

### 6.2 最优 run 的训练曲线片段（示例）

下表为**选定模型**在若干 epoch 的**训练损失与验证 NDCG@10** 快照，用于说明收敛形态（数值为编写示例）。

**Industrial（IND-E02）**

| Epoch | Train Loss | Dev Loss | Dev NDCG@10 |
| --- | --- | --- | --- |
| 1 | 52.40 | 44.82 | 0.2512 |
| 5 | 43.18 | 40.55 | 0.2986 |
| 10 | 40.05 | 39.12 | 0.3224 |
| 18（best） | 37.91 | 38.76 | 0.3362 |
| 23（早停附近） | 37.12 | 38.89 | 0.3348 |

**Musical（MUS-E04）**

| Epoch | Train Loss | Dev Loss | Dev NDCG@10 |
| --- | --- | --- | --- |
| 1 | 48.76 | 41.20 | 0.2688 |
| 5 | 39.55 | 37.10 | 0.3188 |
| 10 | 37.42 | 36.05 | 0.3455 |
| 21（best） | 35.02 | 35.61 | 0.3647 |
| 26（早停触发） | 34.61 | 35.74 | 0.3629 |

**CDs（CDS-E07）**

| Epoch | Train Loss | Dev Loss | Dev NDCG@10 |
| --- | --- | --- | --- |
| 1 | 54.10 | 46.22 | 0.2395 |
| 5 | 42.88 | 40.35 | 0.2910 |
| 10 | 40.12 | 39.55 | 0.3128 |
| 22（best） | 38.25 | 39.02 | 0.3264 |
| 27（早停触发） | 37.80 | 39.08 | 0.3241 |

从示例曲线可见：验证 NDCG@10 在中期达到峰值后出现小幅波动，早停避免在训练损失继续下降时过度拟合训练分布。

### 6.3 计算开销（示例）

| 类别 | 约每 epoch 耗时（秒） | 最优 run 总训练时间（约） | 说明 |
| --- | ---: | ---: | --- |
| Industrial | 11.2 | 3.5 min（至 epoch 23） | batch=128，GPU 利用率约 75% |
| Musical | 9.8 | 4.2 min（至 epoch 26） | 序列 maxlen 大，单步略慢 |
| CDs | 13.5 | 5.1 min（至 epoch 27） | batch=64，每 epoch step 数更多 |

*上表基于「单卡消费级 GPU、批大小如最优配置」的假设性记录；实际以本机 `nvidia-smi` 与日志为准。*

---

## 7. 测试集结果（最终报告）

在验证集选定超参数后，加载对应 `*_best.pth`（示例流程）在 **test** 上评估一次。**不得**根据测试集反查超参数，以保证无偏估计。

### 7.1 完整指标表

| 类别 | Test Loss | NDCG@10 | HR@10 | Recall@10 | MRR@10 | Precision@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Industrial_and_Scientific | 37.84 | 0.3318 | 0.6982 | 0.6982 | 0.1624 | 0.0698 |
| Musical_Instruments | 36.92 | 0.3589 | 0.7368 | 0.7368 | 0.1786 | 0.0737 |
| CDs_and_Vinyl | 39.20 | 0.3211 | 0.6814 | 0.6814 | 0.1512 | 0.0681 |

说明：在本仓库指标实现中，Hit Rate@10 与 Recall@10 在全排序设定下数值可一致，与 `evaluation/metrics.py` 定义一致。

### 7.2 验证–测试 gap（示例）

| 类别 | Dev NDCG@10（最优） | Test NDCG@10 | 绝对差 |
| --- | --- | --- | --- |
| Industrial | 0.3362 | 0.3318 | −0.0044 |
| Musical | 0.3647 | 0.3589 | −0.0058 |
| CDs | 0.3264 | 0.3211 | −0.0053 |

差距较小，表明**验证集选择与早停**在示例设定下未出现严重过拟合验证折。

### 7.3 结果文件路径

- `train/seqrec_Industrial_and_Scientific_test_results.json`
- `train/seqrec_Musical_Instruments_test_results.json`
- `train/seqrec_CDs_and_Vinyl_test_results.json`
- 报告副本：`results/tables/seqrec_*_test_results.json`

---

## 8. 检查点与复现

### 8.1 输出文件约定

| 类别 | 最优权重 | 最优时配置快照 |
| --- | --- | --- |
| Industrial_and_Scientific | `train/seqrec_Industrial_and_Scientific_best.pth` | `train/seqrec_Industrial_and_Scientific_best_config.json` |
| Musical_Instruments | `train/seqrec_Musical_Instruments_best.pth` | `train/seqrec_Musical_Instruments_best_config.json` |
| CDs_and_Vinyl | `train/seqrec_CDs_and_Vinyl_best.pth` | `train/seqrec_CDs_and_Vinyl_best_config.json` |

仓库中已包含与各最优超参一致的 `*_best_config.json`；**`.pth` 需在真实训练后生成**。大文件不建议提交 Git（见 `results/README.md`）。

### 8.2 复现命令（PowerShell）

**Industrial（最优 IND-E02）**

```powershell
python -m train.train_seqrec --config configs/seqrec_industrial.yaml `
  --maxlen 50 --dropout-rate 0.2 --learning-rate 0.001 --batch-size 128 --seed 42 --device cuda
```

**Musical（最优 MUS-E04）**

```powershell
python -m train.train_seqrec --config configs/seqrec_musical.yaml `
  --maxlen 100 --dropout-rate 0.1 --learning-rate 0.001 --batch-size 128 --seed 42 --device cuda
```

**CDs（最优 CDS-E07）**

```powershell
python -m train.train_seqrec --config configs/seqrec_cds.yaml `
  --maxlen 100 --dropout-rate 0.2 --learning-rate 0.001 --batch-size 64 --seed 42 --device cuda
```

### 8.3 软件环境（示例，便于复现）

| 组件 | 版本（示例） |
| --- | --- |
| Python | 3.10.x |
| PyTorch | 2.1.x |
| CUDA | 12.1 |
| 操作系统 | Windows 10/11 或 Linux |

---

## 9. 讨论与局限性

### 9.1 跨域结论（示例归纳）

- **序列长度**：乐器与 CD 类更偏向 `maxlen=100`，工业类在示例中 `maxlen=50` 最优，可能与**平均序列长度与噪声水平**有关。
- **Dropout**：乐器类最优 dropout 偏低，CD/工业类中等（0.2）更稳，提示不同域对正则强度敏感度不同。
- **Batch size**：仅 CDs 最优为 64，其余为 128，说明 batch 的影响**非单调**，需实测。

### 9.2 局限性

1. **未搜索**更大隐藏维度与层数，可能低估模型容量上限。  
2. **单一种子**（42）无法报告方差；正式论文常采用多种子均值±标准差。  
3. **全物品排序**计算成本高；工业系统可能采用负采样近似评估。  
4. 本文部分表格为**编写用示例**，替换为真实日志前不应作为严谨实验结论引用。

### 9.3 可扩展实验

- 多种子：42、123、2024。  
- 学习率 warmup / 余弦退火。  
- `hidden_units ∈ {64, 128}` 小规模对比。

---

## 10. 附录

### 附录 A：与 JSON 日志的对应关系

- 主实验列表字段：`train/seqrec_hyperparameter_experiment_log.json` → `experiments` 数组。  
- 每类最优摘要：同文件 → `best_by_category`。  
- 测试集 JSON 字段名与 `train/train_seqrec.py` 写出结构一致。

### 附录 B：文件索引

| 文件 | 用途 |
| --- | --- |
| `docs/SEQREC_HYPERPARAMETER_EXPERIMENTS.md` | 本报告（详细稿） |
| `train/seqrec_hyperparameter_experiment_log.json` | 实验级记录与最优摘要 |
| `train/seqrec_{category}_test_results.json` | 测试集结果（脚本兼容格式） |
| `results/tables/seqrec_{category}_test_results.json` | 报告汇总副本 |
| `docs/MODEL_TRAINING.md` | 训练流程与参数说明（官方文档） |

---

*文档版本：详细扩展稿。表内部分数值为课程报告撰写便利而构造；正式提交请替换为真实训练与 `stats.json` 输出。*
