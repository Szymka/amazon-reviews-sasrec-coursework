# Training

Train the LLMRank standalone sequential backbone on coursework TSV tensors:

```
train/train_llmrank.py
```

Run from repo root (`amazon-reviews-sasrec-coursework/`):

```powershell
python -m train.train_llmrank --config configs/llmrank_industrial.yaml
```

Check processed data compatibility:

```
train/check_data_loading.py
```
