"""Teach upstream LLMRank ``Rank`` about Amazon coursework dataset names (same prompts/parsing as Games)."""

from __future__ import annotations

import os.path as osp
from typing import Any
AMAZON_COURSEWORK = frozenset(
    {
        "Industrial_and_Scientific",
        "Musical_Instruments",
        "CDs_and_Vinyl",
    }
)


def apply_patch(Rank: type[Any]) -> None:
    """Mutate RecBole Rank class from ``model.rank`` after import."""

    _orig_load_text = Rank.load_text

    def load_text(self: Any) -> list[str]:
        if self.dataset_name not in AMAZON_COURSEWORK:
            return _orig_load_text(self)
        token_text: dict[str, str] = {}
        item_text = ["[PAD]"]
        feat_path = osp.join(self.data_path, f"{self.dataset_name}.item")
        with open(feat_path, "r", encoding="utf-8") as file:
            file.readline()
            for line in file:
                item_id, title = line.strip().split("\t")
                token_text[item_id] = title
        for _i, token in enumerate(self.id_token):
            if token == "[PAD]":
                continue
            raw_text = token_text.get(token, f"item {token}")
            item_text.append(raw_text)
        return item_text

    Rank.load_text = load_text  # type: ignore[assignment]

    _orig_predict_on_subsets = Rank.predict_on_subsets

    def predict_on_subsets(self: Any, interaction: Any, idxs: Any) -> Any:
        prev = self.dataset_name
        if prev in AMAZON_COURSEWORK:
            self.dataset_name = "Games"
        try:
            return _orig_predict_on_subsets(self, interaction, idxs)
        finally:
            self.dataset_name = prev

    Rank.predict_on_subsets = predict_on_subsets  # type: ignore[assignment]
