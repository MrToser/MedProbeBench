# -*- coding: utf-8 -*-
"""Metrics for MedXpertQA benchmark."""
import re

from .._metric_base import MetricBase, MetricResult, MetricType
from .._solution import SolutionOutput


class MedXpertQAAccuracy(MetricBase):
    """Accuracy metric for MedXpertQA benchmark."""

    def __init__(self, ground_truth_label: str) -> None:
        """Initialize the MedXpertQAAccuracy metric.

        Args:
            ground_truth_label (`str`):
                The correct answer label (A-J).
        """
        super().__init__(
            name="accuracy",
            metric_type=MetricType.NUMERICAL,
            description="Accuracy of the answer",
        )
        self.ground_truth_label = ground_truth_label.strip().upper()

    def _extract_answer(self, text: str) -> str | None:
        """Extract the answer letter from the response text.

        Args:
            text (`str`): The response text.

        Returns:
            `str | None`: The extracted answer letter or None.
        """
        if not text:
            return None

        # Try to find single letter patterns like "A", "(A)", "A)", etc.
        patterns = [
            r'\b([A-J])\b',  # Single letter
            r'\(([A-J])\)',  # (A)
            r'([A-J])\)',    # A)
            r'answer is ([A-J])',  # "answer is A"
            r'correct answer is ([A-J])',  # "correct answer is A"
            r'option ([A-J])',  # "option A"
        ]

        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                return match.group(1)

        return None

    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        """Evaluate the solution.

        Args:
            solution (`SolutionOutput`):
                The solution to evaluate.

        Returns:
            `MetricResult`: The evaluation result.
        """
        if not solution.success:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={
                    "error": "Solution execution failed",
                },
            )

        # Extract answer from solution output
        output_text = str(solution.output)
        predicted_label = self._extract_answer(output_text)

        if predicted_label is None:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={
                    "error": "Could not extract answer from output",
                    "output": output_text[:200],  # Limit output length
                },
            )

        # Compare with ground truth
        is_correct = predicted_label == self.ground_truth_label
        score = 1.0 if is_correct else 0.0

        return MetricResult(
            name=self.name,
            result=score,
            metadata={
                "predicted": predicted_label,
                "ground_truth": self.ground_truth_label,
                "is_correct": is_correct,
            },
        )


class MedXpertQACategoryAccuracy(MetricBase):
    """Category-wise accuracy metric for MedXpertQA benchmark."""

    def __init__(
        self,
        ground_truth_label: str,
        category: str,
        category_value: str,
    ) -> None:
        """Initialize the MedXpertQACategoryAccuracy metric.

        Args:
            ground_truth_label (`str`):
                The correct answer label (A-J).
            category (`str`):
                The category name (e.g., 'medical_task', 'body_system').
            category_value (`str`):
                The category value.
        """
        super().__init__(
            name=f"accuracy_{category}",
            metric_type=MetricType.NUMERICAL,
            description=f"Accuracy for {category}: {category_value}",
        )
        self.ground_truth_label = ground_truth_label.strip().upper()
        self.category = category
        self.category_value = category_value

    def _extract_answer(self, text: str) -> str | None:
        """Extract the answer letter from the response text.

        Args:
            text (`str`): The response text.

        Returns:
            `str | None`: The extracted answer letter or None.
        """
        if not text:
            return None

        patterns = [
            r'\b([A-J])\b',
            r'\(([A-J])\)',
            r'([A-J])\)',
            r'answer is ([A-J])',
            r'correct answer is ([A-J])',
            r'option ([A-J])',
        ]

        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                return match.group(1)

        return None

    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        """Evaluate the solution for a specific category.

        Args:
            solution (`SolutionOutput`):
                The solution to evaluate.

        Returns:
            `MetricResult`: The evaluation result.
        """
        if not solution.success:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={
                    "error": "Solution execution failed",
                    "category": self.category,
                    "category_value": self.category_value,
                },
            )

        output_text = str(solution.output)
        predicted_label = self._extract_answer(output_text)

        if predicted_label is None:
            return MetricResult(
                name=self.name,
                result=0.0,
                metadata={
                    "error": "Could not extract answer from output",
                    "category": self.category,
                    "category_value": self.category_value,
                },
            )

        is_correct = predicted_label == self.ground_truth_label
        score = 1.0 if is_correct else 0.0

        return MetricResult(
            name=self.name,
            result=score,
            metadata={
                "predicted": predicted_label,
                "ground_truth": self.ground_truth_label,
                "is_correct": is_correct,
                "category": self.category,
                "category_value": self.category_value,
            },
        )
