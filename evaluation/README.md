# Evaluation

- **Top-K coursework splits**: use `evaluation/evaluate_topk.py` against checkpoints from `train/train_llmrank.py`, e.g. `train/llmrank_<category>_best.pth` plus sibling `llmrank_<category>_best_config.json`.

- Training writes test metrics to `train/llmrank_{category}_test_results.json` and `results/tables/llmrank_{category}_test_results.json` automatically.
