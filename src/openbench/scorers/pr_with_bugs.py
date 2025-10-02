import re
from typing import Callable

from inspect_ai.scorer import accuracy, stderr, scorer, Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model


GRADER_TEMPLATE = """
You are an impartial judge.

Given the following model response, you must return either 0 or 1, enclosed in brackets as <answer>0</answer> or <answer>1</answer>.

If the model correctly identified an error any sort in at least one of the unit test responses, return <answer>1</answer>. Otherwise, return <answer>0</answer>.

Model response:

{model_response}

""".strip()


def _extract_pass_fail(text: str) -> bool:
    """Extract PASS/FAIL from grader response."""
    return re.search(r"<answer>\s*1\s*</answer>", text, re.IGNORECASE) is not None


@scorer(metrics=[accuracy(), stderr()])
def pr_with_bugs_scorer(model: str = "openai/gpt-5") -> Callable:
    """Scorer for PR with Bugs using an LLM judge to detect runtime errors.

    Args:
        model: grader model identifier (default: gpt-5)
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's review response
        model_response = state.output.completion

        # Extract the code from the user message
        # The code is in the user_message from the dataset
        code = ""
        for msg in state.messages:
            if hasattr(msg, "role") and msg.role == "user":
                code = msg.content
                break

        # Format grading prompt
        grader_prompt = GRADER_TEMPLATE.format(
            model_response=model_response,
        )

        message = ChatMessageUser(content=grader_prompt)
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion

        passed = _extract_pass_fail(grading_text)
        value = 1.0 if passed else 0.0

        return Score(
            value=value,
            answer=model_response,
            metadata={
                "passed": passed,
                "grading_response": grading_text,
                "category": state.metadata.get("category") if state.metadata else None,
                "repository": state.metadata.get("repository")
                if state.metadata
                else None,
            },
        )

    return score
