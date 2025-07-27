# harness.py

import re
from data_structures import RegexProblem, RegexSolution

def evaluate_solution(solution: RegexSolution) -> RegexSolution:
    """
    Deterministically evaluates a proposed regex against a problem's constraints.

    This function is the system's "ground truth." It takes no shortcuts and uses
    the standard Python `re` engine to verify correctness.

    Args:
        solution: The RegexSolution object containing the problem and the proposed regex.

    Returns:
        The same RegexSolution object, with its `is_correct` and `failed_on`
        attributes updated based on the evaluation.
    """
    try:
        # Step 1: Check if the regex is valid by compiling it.
        compiled_regex = re.compile(solution.proposed_regex)
    except re.error:
        solution.is_correct = False
        solution.failed_on = ["Invalid Regex Pattern"]
        return solution

    failures = []

    # Step 2: Check for False Negatives (must_match strings that didn't match).
    for text in solution.problem.must_match:
        if not compiled_regex.search(text):
            failures.append(f"False Negative on: '{text}'")

    # Step 3: Check for False Positives (must_not_match strings that did match).
    for text in solution.problem.must_not_match:
        if compiled_regex.search(text):
            failures.append(f"False Positive on: '{text}'")

    if not failures:
        solution.is_correct = True
        solution.failed_on = []
    else:
        solution.is_correct = False
        solution.failed_on = failures

    return solution