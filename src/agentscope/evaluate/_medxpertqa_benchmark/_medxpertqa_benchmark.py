# -*- coding: utf-8 -*-
"""MedXpertQA Benchmark implementation for AgentScope."""
import json
import os
from typing import Generator

from ._medxpertqa_metric import MedXpertQAAccuracy, MedXpertQACategoryAccuracy
from .._benchmark_base import BenchmarkBase
from .._task import Task


class MedXpertQABenchmark(BenchmarkBase):
    """MedXpertQA benchmark for evaluating medical QA capabilities."""

    def __init__(
        self,
        data_path: str,
        sample_ratio: float = 1.0,
    ) -> None:
        """Initialize the MedXpertQABenchmark.

        Args:
            data_path (`str`):
                Path to the JSONL data file.
            sample_ratio (`float`, defaults to `1.0`):
                Ratio of samples to use (0.0-1.0). Samples are selected
                using stratified sampling by category with fixed intervals.
        """
        super().__init__(
            name="MedXpertQA",
            description="Medical expert QA benchmark for evaluating "
            "medical knowledge and reasoning capabilities.",
        )

        self.data_path = os.path.abspath(data_path)
        self.sample_ratio = max(0.0, min(1.0, sample_ratio))

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found at: {self.data_path}",
            )

        if not self.data_path.endswith(".jsonl"):
            raise ValueError(
                "Data file must be in JSONL format (.jsonl)",
            )

        self.dataset = self._load_data()

    def _load_data(self) -> list[dict]:
        """Load the dataset from the JSONL file.

        Returns:
            `list[dict]`: List of data items.
        """
        dataset = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    dataset.append(data)

        # Apply stratified sampling if sample_ratio < 1.0
        if self.sample_ratio < 1.0:
            dataset = self._stratified_sample(dataset)

        return dataset

    def _stratified_sample(self, dataset: list[dict]) -> list[dict]:
        """Apply stratified sampling by category.

        Args:
            dataset (`list[dict]`): Full dataset.

        Returns:
            `list[dict]`: Sampled dataset.
        """
        # Group by combined key: medical_task|body_system|question_type
        categories = {}
        for item in dataset:
            medical_task = item.get("medical_task", "Unknown")
            body_system = item.get("body_system", "Unknown")
            question_type = item.get("question_type", "Unknown")
            category = f"{medical_task}|{body_system}|{question_type}"
            if category not in categories:
                categories[category] = []
            categories[category].append(item)

        # Sample from each category with fixed interval
        sampled = []
        for category_items in categories.values():
            n_items = len(category_items)
            n_sample = max(1, int(n_items * self.sample_ratio))
            if n_sample >= n_items:
                sampled.extend(category_items)
            else:
                # Fixed interval sampling
                interval = n_items / n_sample
                indices = [int(i * interval) for i in range(n_sample)]
                sampled.extend([category_items[i] for i in indices])

        return sampled

    @staticmethod
    def _data_to_task(item: dict) -> Task:
        """Convert a dataset item to a Task object.

        Args:
            item (`dict`): A data item from the dataset.

        Returns:
            `Task`: A Task object for evaluation.
        """
        # Format the question with options
        question_text = item["question"]
        options = item["options"]

        # Build formatted question
        formatted_question = f"{question_text}\n\nOptions:\n"
        for key, value in sorted(options.items()):
            formatted_question += f"({key}) {value}\n"
        formatted_question += (
            "\nPlease answer with the correct option letter (A-J)."
        )

        return Task(
            id=item["id"],
            input=formatted_question,
            ground_truth={
                "label": item["label"],
                "medical_task": item.get("medical_task"),
                "body_system": item.get("body_system"),
                "question_type": item.get("question_type"),
            },
            tags={
                "medical_task": item.get("medical_task", "Unknown"),
                "body_system": item.get("body_system", "Unknown"),
                "question_type": item.get("question_type", "Unknown"),
            },
            metrics=[
                MedXpertQAAccuracy(item["label"]),
                MedXpertQACategoryAccuracy(
                    item["label"],
                    "medical_task",
                    item.get("medical_task", "Unknown"),
                ),
                MedXpertQACategoryAccuracy(
                    item["label"],
                    "body_system",
                    item.get("body_system", "Unknown"),
                ),
                MedXpertQACategoryAccuracy(
                    item["label"],
                    "question_type",
                    item.get("question_type", "Unknown"),
                ),
            ],
            metadata={
                "options": options,
                "original_question": item["question"],
            },
        )

    def __iter__(self) -> Generator[Task, None, None]:
        """Iterate over the benchmark.

        Yields:
            `Task`: A Task object for each item in the dataset.
        """
        for item in self.dataset:
            yield self._data_to_task(item)

    def __getitem__(self, index: int) -> Task:
        """Get a task by index.

        Args:
            index (`int`): The index of the task.

        Returns:
            `Task`: The task at the given index.
        """
        return self._data_to_task(self.dataset[index])

    def __len__(self) -> int:
        """Get the length of the benchmark.

        Returns:
            `int`: The number of tasks in the benchmark.
        """
        return len(self.dataset)
