# Data Download

## 数据来源

本项目使用 Amazon Reviews 2023 5-Core 数据，面向推荐系统课程大作业中的序列推荐实验。完整数据体积较大，不上传到 GitHub；每个成员需要在本地下载或通过小组私有网盘获取。

## 使用类别

本作业需要三个商品类别：

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

## 每个类别需要的文件

每个类别需要 5 个官方文件：

```text
<category>.train.csv.gz
<category>.valid.csv.gz
<category>.test.csv.gz
<category>.jsonl.gz
meta_<category>.jsonl.gz
```

说明：

- `<category>.jsonl.gz` 是 User Review 文件。
- `meta_<category>.jsonl.gz` 是 Item Metadata 文件。
- raw 阶段官方验证集文件名为 `valid.csv.gz`；预处理后输出为 `dev.tsv`。

## 推荐放置路径

```text
data/raw/Industrial_and_Scientific/
data/raw/Musical_Instruments/
data/raw/CDs_and_Vinyl/
```

示例：

```text
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.train.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.valid.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.test.csv.gz
data/raw/Industrial_and_Scientific/Industrial_and_Scientific.jsonl.gz
data/raw/Industrial_and_Scientific/meta_Industrial_and_Scientific.jsonl.gz
```

## 检查 raw 数据

先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

检查文件是否齐全、非 0 字节、CSV 字段是否正确、JSONL 是否可读取：

```powershell
python scripts/check_raw_data.py
```

Windows PowerShell 如果找不到 `python`，可使用本机绝对路径：

```powershell
$env:PYTHONIOENCODING='utf-8'; & "C:\Users\Chess\AppData\Local\Programs\Python\Python312\python.exe" scripts/check_raw_data.py
```

该脚本只读取少量样本行，不下载数据，不执行预处理。
