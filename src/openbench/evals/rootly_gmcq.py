"""
GitHub Multiple Choice Questions
Authored by:
Rootly AI Labs
Based on: https://huggingface.co/datasets/TheFloatingString/gmcq

# run code generation
bench eval gmcq --model "groq/llama-3.1-8b-instant" --T subtask=mastodon

If subtask is None, then the entire dataset is used.

Please refer to https://huggingface.co/datasets/TheFloatingString/gmcq for the subtask to use.
There are 6 subtasks as of Tuesday, August 19, 2025, and the None option for the entire dataset:

- bluesky
- chroma
- cloudflare
- duckdb
- mastodon
- tailscale
- None
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.rootly_gmcq import load_rootly_gmcq_dataset
from openbench.scorers.mcq import simple_mcq_scorer
from inspect_ai.solver import generate


@task
def rootly_gmcq(subtask: str = None) -> Task:  # type: ignore
    dataset = load_rootly_gmcq_dataset(subtask)
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=simple_mcq_scorer(),
        config=GenerateConfig(),
    )
