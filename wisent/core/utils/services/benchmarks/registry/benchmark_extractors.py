"""
Answer extractors for TaskInterface benchmarks.

These extractors parse model outputs to extract answers for validation.
Different from LMEvalBenchmarkExtractor which creates contrastive pairs.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re

from wisent.core.utils.infra_tools.errors import NumericalExtractionError, TextExtractionError, ExtractorNotFoundError

# Re-export from helpers
from wisent.core.utils.services.benchmarks._helpers.benchmark_extractors_helpers import SuperGPQAExtractor
from wisent.core.utils.config_tools.constants import COMPARE_TOL


class BenchmarkExtractor(ABC):
    """Base class for benchmark answer extraction."""

    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from model's generated text."""
        pass

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        return answer.lower().strip()

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        return self.normalize_answer(predicted) == self.normalize_answer(expected)

    def extract_qa_pair(self, sample: dict, task: any = None) -> Optional[dict]:
        """Extract a question-answer pair from a sample dictionary."""
        question = None
        answer = None

        for q_field in ["question", "prompt", "text", "input", "problem", "query"]:
            if q_field in sample and sample[q_field]:
                question = str(sample[q_field]).strip()
                break

        for a_field in ["answer", "target", "label", "output", "solution", "expected"]:
            if a_field in sample and sample[a_field]:
                answer = str(sample[a_field]).strip()
                if "####" in answer:
                    answer = answer.split("####")[-1].strip()
                break

        if not question or not answer:
            return None

        return {
            "formatted_question": f"Question: {question}",
            "correct_answer": answer,
        }

    def extract_contrastive_pair(self, sample: dict, task: any = None) -> Optional[dict]:
        """Extract a contrastive pair (question, correct, incorrect) from a sample."""
        qa_pair = self.extract_qa_pair(sample, task)
        if not qa_pair:
            return None

        correct = qa_pair["correct_answer"]
        try:
            num_val = float(correct.replace(',', ''))
            incorrect_val = num_val + 1
            incorrect = str(int(incorrect_val)) if incorrect_val == int(incorrect_val) else str(incorrect_val)
        except (ValueError, TypeError):
            incorrect = "I don't know"

        return {
            "question": qa_pair["formatted_question"],
            "correct_answer": correct,
            "incorrect_answer": incorrect,
        }


class GSM8KExtractor(BenchmarkExtractor):
    """Extractor for GSM8K and math tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from text."""
        if not text:
            return None

        try:
            import json
            if "{" in text and "}" in text:
                json_match = re.search(r'\{[^}]*"final_answer"[^}]*\}', text)
                if json_match:
                    data = json.loads(json_match.group(0))
                    answer = data.get("final_answer")
                    if answer:
                        answer = re.sub(r'[^\d.\-]', '', str(answer))
                        if answer and answer.replace('.', '').replace('-', '').isdigit():
                            return answer
        except Exception:
            pass

        hash_match = re.search(r'####\s*([\d,.\-]+)', text)
        if hash_match:
            return hash_match.group(1).replace(',', '')

        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,.\-]+)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,.\-]+)',
            r'=\s*([\d,.\-]+)\s*$',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')

        raise NumericalExtractionError(response=text)

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Compare numerical answers with tolerance."""
        if predicted is None:
            return False
        try:
            return abs(float(predicted) - float(expected)) < COMPARE_TOL
        except (ValueError, TypeError):
            return self.normalize_answer(predicted) == self.normalize_answer(expected)


class LiveCodeBenchExtractor(BenchmarkExtractor):
    """Extractor for coding tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract code from model response."""
        if not text:
            return None

        code_block_patterns = [r'```python\s*(.*?)```', r'```\s*(.*?)```', r'`([^`]+)`']
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return max(matches, key=len).strip()

        func_match = re.search(r'(def\s+\w+.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()
        class_match = re.search(r'(class\s+\w+.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if class_match:
            return class_match.group(1).strip()
        if 'def ' in text or 'class ' in text or 'import ' in text:
            return text.strip()
        return None

    def check_answer(self, predicted: str, expected: str) -> bool:
        if predicted is None:
            return False
        return len(predicted.strip()) > 0


class HLEExtractor(BenchmarkExtractor):
    """Extractor for HLE tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        if not text:
            return None
        try:
            import json
            if "{" in text and "}" in text:
                json_match = re.search(r'\{[^}]*"answer"[^}]*\}', text)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return str(data.get("answer", ""))
        except Exception:
            pass
        answer_match = re.search(r'answer\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        mc_match = re.search(r'\b([A-D])\b', text)
        if mc_match:
            return mc_match.group(1)
        raise TextExtractionError(response=text)

    def check_answer(self, predicted: str, expected: str) -> bool:
        if predicted is None:
            return False
        pred_norm = self.normalize_answer(predicted)
        exp_norm = self.normalize_answer(expected)
        return exp_norm in pred_norm or pred_norm in exp_norm


class AxBenchExtractor(BenchmarkExtractor):
    """Extractor for AxBench concept-steering tasks (pyvene/axbench-concept*).

    Generations are free-form steered text scored by the AxBench LLM judge
    (concept/instruct/fluency harmonic mean), so answer extraction is the
    identity on the generated text. Contrastive pairs map the dataset's
    instruction (``input``) to the concept-incorporating response
    (``output``); the incorrect side is supplied by the concept-level pairing
    in tasks/concepts/axbench_pairs.py, not per-sample.
    """

    def extract_answer(self, text: str) -> Optional[str]:
        """Return the generated text unchanged (judge-scored task)."""
        if not text:
            return None
        return text.strip()

    def extract_qa_pair(self, sample: dict, task: any = None) -> Optional[dict]:
        question = str(sample.get("input", "")).strip()
        answer = str(sample.get("output", "")).strip()
        if not question or not answer:
            return None
        return {
            "formatted_question": question,
            "correct_answer": answer,
        }

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Judge-scored task: any non-empty generation is extractable."""
        return bool(predicted and predicted.strip())


# Registry mapping task names to extractors
_EXTRACTOR_REGISTRY = {}


def _init_registry():
    """Initialize the registry using helper."""
    from wisent.core.utils.services.benchmarks._helpers.benchmark_extractors_helpers import _populate_registry
    _populate_registry(_EXTRACTOR_REGISTRY)


_init_registry()


def get_extractor(task_name: str) -> BenchmarkExtractor:
    """Get the appropriate extractor for a task."""
    from wisent.core.utils.services.benchmarks._helpers.benchmark_extractors_helpers import (
        get_extractor as _get_extractor,
    )
    return _get_extractor(task_name, _EXTRACTOR_REGISTRY)
