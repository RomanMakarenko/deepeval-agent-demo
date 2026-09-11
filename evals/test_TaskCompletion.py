import os
import sys

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_instrumented import support_agent
from local_models import judge_model


# Set DEEPEVAL_JUDGE_PROVIDER=ollama in .env for local evaluation.

actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)

evaluate(test_cases = [test_case],
         metrics=[TaskCompletionMetric(threshold=0.7, model=judge_model)])