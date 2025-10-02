"""
PR with Bugs Benchmark
A benchmark for evaluating model ability to review GitHub pull requests and identify runtime errors
Based on: https://huggingface.co/datasets/TheFloatingString/github_assistant

# Run evaluation
bench eval pr_with_bugs --model "groq/llama-3.1-8b-instant"

# Filter by category
bench eval pr_with_bugs --model "groq/llama-3.1-8b-instant" --T category=indentation

# Filter by repository
bench eval pr_with_bugs --model "groq/llama-3.1-8b-instant" --T repository=django

# Use custom grader model
bench eval pr_with_bugs --model "groq/llama-3.1-8b-instant" --T grader_model=openai/gpt-4o-mini

Available categories: indentation, etc.
Available repositories: django, etc.
"""

from typing import Optional
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate
from openbench.scorers.pr_with_bugs import pr_with_bugs_scorer


def record_to_sample(record: dict) -> Sample:
    """Convert a GitHub Assistant record to an Inspect Sample.

    Args:
        record: Dataset record containing system_prompt, user_message, category, repository

    Returns:
        Sample with formatted input messages
    """
    system_prompt = record.get("system_prompt", "")
    user_message = record.get("user_message", "")

    # Build chat messages with system prompt and user message
    messages = []
    if system_prompt:
        messages.append(ChatMessageSystem(content=system_prompt))
    messages.append(ChatMessageUser(content=user_message))

    return Sample(
        input=messages,
        target="",  # No target needed for model-graded evaluation
        metadata={
            "category": record.get("category"),
            "repository": record.get("repository"),
        },
    )


@task
def pr_with_bugs(
    category: Optional[str] = None,
    repository: Optional[str] = None,
    grader_model: str = "openai/gpt-4o-mini",
) -> Task:  # type: ignore
    """GitHub PR review evaluation using the github_assistant dataset.

    Uses a model grader to evaluate if the model correctly identifies runtime errors.

    Args:
        category: Optional filter by category (e.g., "indentation")
        repository: Optional filter by repository (e.g., "django")
        grader_model: Model to use for grading (default: openai/gpt-4o-mini)

    Returns:
        Task configured for GitHub PR review evaluation
    """

    # Filter by category and/or repository if provided
    def mapper_with_filter(record: dict) -> Sample | list[Sample]:
        rec_category = record.get("category")
        rec_repository = record.get("repository")

        if (category is None or rec_category == category) and (
            repository is None or rec_repository == repository
        ):
            return record_to_sample(record)
        else:
            return []

    dataset = hf_dataset(
        "TheFloatingString/github_assistant",
        split="train",
        sample_fields=mapper_with_filter,
        auto_id=True,
    )

    return Task(
        name="pr_with_bugs",
        dataset=dataset,
        solver=[generate()],
        scorer=pr_with_bugs_scorer(model=grader_model),
        config=GenerateConfig(),
    )
