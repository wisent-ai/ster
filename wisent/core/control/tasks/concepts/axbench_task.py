"""AxBench concept-steering task (pyvene/axbench-concept* datasets).

AxBench (arXiv 2501.17148) evaluates steering methods per concept: the train
split provides concept-incorporating positives and plain instruction
negatives for training a per-concept steering direction; the test split
provides labelled passages for concept detection. Columns: input, output,
output_concept, concept_genre, category, dataset_category, concept_id.
"""

from typing import Any, Dict, List, Optional

import datasets

from wisent.core.control.tasks.base.task_interface import TaskInterface
from wisent.core.utils.services.benchmarks import AxBenchExtractor

__all__ = ["AxBenchTask", "DATASET_CONFIGS"]

DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "concept500": {"source": "pyvene/axbench-concept500"},
    "concept16k": {"source": "pyvene/axbench-concept16k"},
    "concept16k_v2": {"source": "pyvene/axbench-concept16k_v2"},
}

# Concept sets are released per GemmaScope subject model/layer; each subset
# directory holds train/data.parquet and test/data.parquet.
SUBSETS = ["2b/l10", "2b/l20", "9b/l20", "9b/l31"]
DEFAULT_SUBSET = "2b/l10"

_NEGATIVE_CONCEPT_ID = -1
_CATEGORY_POSITIVE = "positive"
_CATEGORY_NEGATIVE = "negative"
_CATEGORY_HARD_NEGATIVE = "hard negative"


class AxBenchTask(TaskInterface):
    """AxBench concept dataset with per-concept row accessors."""

    def __init__(
        self,
        variant: str = "concept500",
        subset: str = DEFAULT_SUBSET,
        limit: Optional[int] = None,
        **_: Any,
    ):
        if variant not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown AxBench variant '{variant}'. "
                f"Available: {sorted(DATASET_CONFIGS)}"
            )
        if subset not in SUBSETS:
            raise ValueError(
                f"Unknown AxBench subset '{subset}'. Available: {SUBSETS} "
                "(GemmaScope subject model/layer the concept set was built for)."
            )
        self.variant = variant
        self.subset = subset
        self.source = DATASET_CONFIGS[variant]["source"]
        self._limit = limit
        self._splits: Dict[str, List[Dict[str, Any]]] = {}
        self._extractor = AxBenchExtractor()

    def _split_rows(self, split: str) -> List[Dict[str, Any]]:
        if split not in self._splits:
            dataset = datasets.load_dataset(
                self.source,
                data_files={split: f"{self.subset}/{split}/data.parquet"},
                split=split,
            )
            self._splits[split] = [dict(item) for item in dataset]
        return self._splits[split]

    # -- TaskInterface ----------------------------------------------------

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = self._split_rows("train")
        effective_limit = limit or self._limit
        if effective_limit:
            return rows[:effective_limit]
        return rows

    def get_extractor(self) -> AxBenchExtractor:
        return self._extractor

    def get_name(self) -> str:
        return f"axbench_{self.variant}"

    def get_description(self) -> str:
        return (
            f"AxBench {self.variant}: concept steering and detection benchmark "
            f"({self.source}), judge-scored per the AxBench protocol"
        )

    def get_categories(self) -> List[str]:
        return ["concept-steering", "interpretability", "text_generation"]

    # -- Concept-level accessors ------------------------------------------

    def list_concept_ids(self, split: str = "train") -> List[int]:
        ids = {
            int(row["concept_id"])
            for row in self._split_rows(split)
            if int(row["concept_id"]) != _NEGATIVE_CONCEPT_ID
        }
        return sorted(ids)

    def concept_label(self, concept_id: int) -> str:
        for row in self._split_rows("train"):
            if int(row["concept_id"]) == concept_id and row["category"] == _CATEGORY_POSITIVE:
                return str(row["output_concept"])
        raise ValueError(
            f"No positive rows for concept_id={concept_id} in {self.source} train split."
        )

    def positive_rows(self, concept_id: int, split: str = "train") -> List[Dict[str, Any]]:
        rows = [
            row for row in self._split_rows(split)
            if int(row["concept_id"]) == concept_id and row["category"] == _CATEGORY_POSITIVE
        ]
        if not rows:
            raise ValueError(
                f"No positive rows for concept_id={concept_id} in {self.source} "
                f"{split} split."
            )
        return rows

    def negative_rows(
        self, split: str = "train", concept_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Plain (non-concept) rows. The train split pools negatives under
        concept_id=-1; the test split labels negatives per concept."""
        target = _NEGATIVE_CONCEPT_ID if concept_id is None else concept_id
        rows = [
            row for row in self._split_rows(split)
            if row["category"] == _CATEGORY_NEGATIVE
            and int(row["concept_id"]) == target
        ]
        if not rows:
            raise ValueError(
                f"No negative rows (category={_CATEGORY_NEGATIVE!r}, "
                f"concept_id={target}) in {self.source} {split} split."
            )
        return rows

    def hard_negative_rows(self, concept_id: int, split: str = "train") -> List[Dict[str, Any]]:
        rows = [
            row for row in self._split_rows(split)
            if int(row["concept_id"]) == concept_id
            and row["category"] == _CATEGORY_HARD_NEGATIVE
        ]
        if not rows:
            observed = sorted({row["category"] for row in self._split_rows(split)})
            raise ValueError(
                f"No hard-negative rows (category={_CATEGORY_HARD_NEGATIVE!r}) for "
                f"concept_id={concept_id} in {self.source} {split} split. "
                f"Observed category values: {observed}"
            )
        return rows

    def detection_rows(self, concept_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """Test-split passages for concept detection.

        Returns the concept's positives, its own negatives, and
        "extra_negative" — other concepts' plain negatives, used to build the
        imbalanced evaluation setting.
        """
        extra = [
            row for row in self._split_rows("test")
            if row["category"] == _CATEGORY_NEGATIVE
            and int(row["concept_id"]) != concept_id
        ]
        return {
            "positive": self.positive_rows(concept_id, split="test"),
            "negative": self.negative_rows(split="test", concept_id=concept_id),
            "extra_negative": extra,
        }
