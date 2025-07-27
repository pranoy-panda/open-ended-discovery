# open-ended-discovery/src/llm_agents.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
from data_structures import RegexProblem

# --- Model Loading (as per your snippet) ---
# NOTE: This requires a machine with sufficient RAM/VRAM and the 'transformers'
#       'torch', and 'accelerate' libraries installed.
print("Loading Qwen/Qwen3-4B model... This may take a moment.")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    device_map="auto" , # Automatically uses GPU if available
    load_in_8bit=True,  # or load_in_4bit=True
    torch_dtype=torch.float16
)
print("Model loaded successfully.")


def ask_qwen(prompt: str, max_retries=3) -> str:
    """
    A robust function to send a prompt to the Qwen model and get a response.
    Includes basic chat templating and retry logic.
    """
    for attempt in range(max_retries):
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant that provides concise and accurate responses in the requested format."},
                        {"role": "user", "content": prompt}]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Error during model generation (attempt {attempt+1}): {e}")
            if attempt + 1 == max_retries:
                return "" # Return empty if all retries fail
    return ""


class ProblemGenerator:
    """
    The agent responsible for creating and certifying new regex problems.
    This agent embodies the "Environment" side of the POET algorithm.
    """
    def generate_initial_problem(self) -> Optional[RegexProblem]:
        """Generates the very first problem to kickstart the process."""
        prompt = """
        You are a puzzle designer. Create a simple 'level 1' regular expression problem.
        The problem must only require basic concepts like literal characters, basic character classes (like \\d), and simple quantifiers (*, +).

        Provide your output as a single JSON object with these exact keys:
        "description": A clear English description of the rule.
        "must_match": A list of 5 strings that follow the rule.
        "must_not_match": A list of 5 strings that break the rule.
        "concepts": A list containing only the string 'basic'.
        "level": The integer 1.
        """
        response = ask_qwen(prompt)
        try:
            data = json.loads(response)
            return RegexProblem(**data)
        except (json.JSONDecodeError, TypeError):
            print("ERROR: Failed to decode initial problem JSON from LLM.")
            print("=="*20)
            print(f"Generated response: \n {response}")
            print("=="*20)
            return None

    def mutate_problem(self, seed_problem: RegexProblem) -> Optional[RegexProblem]:
        """
        Creates a new problem by mutating an existing one, making it more complex.
        This is a core step in the POET (Paired Open-Ended Trailblazer) methodology.
        """
        prompt = f"""
        You are a puzzle designer. Your task is to evolve a regular expression problem to make it more challenging.

        Here is the original problem:
        - Description: {seed_problem.description}
        - Level: {seed_problem.level}
        - Concepts Used: {seed_problem.concepts}

        Now, create a new 'level {seed_problem.level + 1}' problem by adding a new constraint or concept.
        For example, you could introduce:
        - A specific quantifier range (e.g., {{3,5}})
        - Grouping with parentheses `()`
        - A negative lookahead `(?!...)` to exclude a pattern.

        Provide your output as a single JSON object with these exact keys:
        "description": The new, more complex description.
        "must_match": A new list of 5 strings for the new rule.
        "must_not_match": A new list of 5 strings for the new rule.
        "concepts": A list of all concepts the new problem uses (e.g., ["basic", "lookaround"]).
        "level": The integer {seed_problem.level + 1}.
        """
        response = ask_qwen(prompt)
        try:
            # The LLM response might be wrapped in markdown, so we extract the JSON
            json_str = response[response.find('{'):response.rfind('}')+1]
            data = json.loads(json_str)
            return RegexProblem(**data)
        except (json.JSONDecodeError, TypeError):
            print("ERROR: Failed to decode mutated problem JSON from LLM.")
            return None


class RegexSolver:
    """
    The agent responsible for solving the problems generated by the ProblemGenerator.
    """
    def solve_problem(self, problem: RegexProblem, few_shot_examples: List = []) -> str:
        """
        Attempts to generate a regex solution for a given problem.
        Uses few-shot learning by incorporating examples of previously solved problems.
        """
        example_str = "Here are some examples of problems you have solved correctly:\n"
        for ex in few_shot_examples:
            example_str += f"- Description: {ex['problem_desc']}\n  Regex: {ex['regex']}\n"

        prompt = f"""
        You are a regular expression expert. Your task is to write a single, Python-compatible regex pattern to solve the following problem.

        {example_str if few_shot_examples else ""}

        **New Problem to Solve:**
        - Description: {problem.description}
        - Must Match: {problem.must_match}
        - Must Not Match: {problem.must_not_match}

        Provide only the raw regex pattern as your answer. Do not include any explanation or code fences.
        """
        return ask_qwen(prompt)

    def debug_problem(self, problem: RegexProblem, failed_regex: str, errors: List[str]) -> str:
        """
        Attempts to fix an incorrect regex based on specific failure feedback.
        """
        prompt = f"""
        You are a regular expression expert. Your previous attempt to solve a problem was incorrect.

        **Problem:**
        - Description: {problem.description}

        **Your Incorrect Regex:**
        `{failed_regex}`

        **Reason for Failure:**
        It failed on the following examples: {errors}

        Please analyze your mistake and provide a new, corrected regex pattern.
        Provide only the raw regex pattern as your answer.
        """
        return ask_qwen(prompt)