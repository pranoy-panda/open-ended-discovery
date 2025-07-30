# open-ended-discovery/src/llm_agents.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List, Union
from dotenv import load_dotenv
from pathlib import Path
import time

from data_structures import RegexProblem, RegexSolution
from harness import evaluate_solution

# Go one folder up and point to the .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
sleep_time = 12 # secs

# --- Model Loading Configuration ---
class ModelConfig:
    """Configuration class to handle different model types."""
    
    def __init__(self, model_type: str = "huggingface", model_name: str = "Qwen/Qwen3-4B", **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.config = kwargs

# --- Hugging Face Model Loading ---
def load_hf_model(model_name: str = "Qwen/Qwen3-4B", **kwargs):
    """
    Load any Hugging Face model with configurable parameters.
    """
    print(f"Loading {model_name} model... This may take a moment.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Default parameters with override capability
    default_params = {
        "device_map": "auto",
        "load_in_8bit": True,
        "torch_dtype": torch.float16
    }
    default_params.update(kwargs)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **default_params)
    print("Model loaded successfully.")
    return tokenizer, model

def ask_hf_model(prompt: str, tokenizer, model, max_retries=3) -> str:
    """
    A robust function to send a prompt to any Hugging Face model and get a response.
    Includes basic chat templating and retry logic.
    """
        
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides concise and accurate responses in the requested format."},
                {"role": "user", "content": prompt}
            ]
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
            print(f"Error during HF model generation (attempt {attempt+1}): {e}")
            if attempt + 1 == max_retries:
                return ""  # Return empty if all retries fail
    return ""

def ask_gemini(prompt: str, llm, max_retries=3) -> str:
    """
    A robust function to send a prompt to Gemini and get a response.
    Includes retry logic.
    """ 
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            time.sleep(sleep_time)
            return response.content.strip()
        except Exception as e:
            print(f"Error during Gemini generation (attempt {attempt+1}): {e}")
            if attempt + 1 == max_retries:
                return ""  # Return empty if all retries fail
    return ""

class ProblemGenerator:
    """
    The agent responsible for creating and certifying new regex problems.
    This agent embodies the "Environment" side of the POET algorithm.
    """
    MAX_CERTIFICATION_ATTEMPTS = 3 # Number of times to try self-correction
    
    def __init__(self, model_type: str = "huggingface", model_name = "Qwen/Qwen3-4B", **model_kwargs):
        """
        Initialize the ProblemGenerator with a specific model type.
        
        Args:
            model_type: Either "huggingface" or "gemini"
            **model_kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        if self.model_type=="huggingface":
            self.hf_tokenizer, self.hf_model = load_hf_model(model_name)
        elif self.model_type=="gemini":
            self.gemini_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=1.0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )

    def _debug_generated_problem(self, problem: RegexProblem, failed_regex: str, errors: list) -> tuple[Optional[RegexProblem], Optional[str]]:
        """An internal method to ask the LLM to fix its own flawed generation."""
        prompt = f"""
        You are a puzzle designer. You previously generated a problem, but your own proposed 'certification_regex' for it was incorrect.

        **The Flawed Problem You Generated:**
        - Description: {problem.description}
        - Test Cases: {problem.must_match} / {problem.must_not_match}

        **Your Incorrect Regex:**
        `{failed_regex}`

        **Reason for Failure:**
        Your regex failed on your own test cases: {errors}

        **YOUR TASK:**
        Analyze your mistake and fix it. You can either fix the 'certification_regex' OR modify the 'description' and test cases to match your regex.
        Provide the complete, corrected problem as a single JSON object, including the new 'certification_regex'.
        """
        response = self._ask_model(prompt)
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            data = json.loads(json_str)
            cert_regex = data.pop("certification_regex", "")
            new_problem = RegexProblem(**data)
            return new_problem, cert_regex
        except (json.JSONDecodeError, TypeError):
            print("ERROR: Failed to decode the debugged problem JSON.")
            return None, None
         
    def _ask_model(self, prompt: str) -> str:
        """Internal method to ask the configured model."""
        return self.ask_llm(prompt, self.model_type, **self.model_kwargs)
    
    def _generate_and_certify(self, prompt: str, is_debug: bool = False) -> Optional[tuple[RegexProblem, str]]:
        """
        A refactored, robust internal method that handles the entire generate-and-certify loop.
        """
        for i in range(self.MAX_CERTIFICATION_ATTEMPTS):
            # On debug attempts, the prompt is constructed by the calling method
            if is_debug and i > 0:
                print(f"GENERATOR: Self-correction attempt {i+1}...")
                response = self._ask_model(prompt) # The debug prompt is passed in
            else: # First attempt
                response = self._ask_model(prompt)
            
            if not response:
                print(f"WARN: LLM provided no response on attempt {i+1}.")
                continue

            try:
                json_str = response[response.find('{'):response.rfind('}')+1]
                data = json.loads(json_str)
                data.pop("simplification_strategy", "")
                cert_regex = data.pop("certification_regex", "")
                problem = RegexProblem(**data)

                cert_solution = evaluate_solution(RegexSolution(problem=problem, proposed_regex=cert_regex))
                if cert_solution.is_correct:
                    print("GENERATOR: Problem certified successfully.")
                    return problem, cert_regex
                else:
                    print(f"GENERATOR: Certification failed. Errors: {cert_solution.failed_on}")
                    # For debug retries, we need to reconstruct the prompt with the new errors
                    if is_debug:
                        prompt = self._create_debug_prompt(problem, cert_regex, cert_solution.failed_on)

            except (json.JSONDecodeError, TypeError):
                 print(f"ERROR: Invalid JSON in generation (Attempt {i+1}).")

        print("WARN: Generator failed to create a valid problem after all attempts.")
        return None, None

    def ask_llm(self, prompt: str, model_type: str = "huggingface", **kwargs) -> str:
        """
        Unified interface to ask any LLM model.
        
        Args:
            prompt: The prompt to send
            model_type: Either "huggingface" or "gemini"
            **kwargs: Additional parameters for specific models
        """
        if model_type.lower() == "huggingface":
            return ask_hf_model(prompt=prompt,tokenizer=self.hf_tokenizer,model=self.hf_model, **kwargs)
        elif model_type.lower() == "gemini":
            return ask_gemini(prompt=prompt,llm=self.gemini_llm, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def generate_initial_problem(self) -> Optional[tuple[RegexProblem, str]]:
        """Generates and certifies the very first problem."""
        print("GENERATOR: Attempting to generate a certified seed problem...")
        prompt = """
            You are a puzzle designer. Create a simple 'level 1' regular expression problem.
            The problem must only require basic concepts like literal characters and basic character classes (like \\d).
            Provide your output as a single JSON object with these exact keys:
            "must_match", "must_not_match", "certification_regex" (the correct regex for your problem/description), "description", "concepts" (as a list), "level" (as an integer).
            """
        return self._generate_and_certify(prompt)

    def generate_next_problem(self, successful_solutions: list, failed_solutions: list, recent_problems: list) -> Optional[tuple[RegexProblem, str]]:
        """
        Generates and certifies the next problem in the curriculum using a robust,
        multi-attempt self-correction loop.
        """
        print("GENERATOR: Attempting to generate a certified next problem...")
        
        # This is the base prompt used for the first attempt.
        base_prompt = f"""
        You are a creative AI curriculum designer. Your goal is to generate the next regular expression problem.

        Analyze all the history to inform your decision:

        **PAST SUCCESSES (The Solver's Strengths):**
        {chr(10).join([f"- Solved: '{p.description}' (Concepts: {p.concepts})" for p in successful_solutions[-5:]])}

        **PAST FAILURES (The Solver's Weaknesses):**
        {chr(10).join([f"- Failed on: '{s.problem.description}'" for s in failed_solutions[-5:]])}

        **RECENTLY GENERATED PROBLEMS (CRITICAL: Do NOT create a similar problem to these!):**
        {chr(10).join([f"- '{desc}'" for desc in recent_problems[-5:]])}

        **YOUR TASK:**
        Based on all the history, devise a completely NEW problem that is distinct from the 'Recently Generated' list. Please note: if in the past there has been very few or no success, please generate a bit simpler problems that before. We want the solver to learn over time. HOWEVER, IF IN THE PAST THERE HAVE BEEN FEW FAILURES, THEN MAKE THE PROBLEMS VERY COMPLEX

        Provide your output as a single JSON object with these exact keys:
            "must_match", "must_not_match", "certification_regex" (the correct regex for your problem/description), "description", "concepts" (as a list), "level" (as an integer).
        """
        return self._generate_and_certify(base_prompt)

    def simplify_problem(self, hard_problem: RegexProblem, errors: list) -> Optional[tuple[RegexProblem, str]]:
        """
        Takes a problem the solver failed on (even with a hint) and makes it simpler.
        This creates a "stepping stone" to bridge a large conceptual gap.
        """
        print("GENERATOR: Attempting to simplify a hard problem...")
        prompt = f"""
        You are an expert curriculum designer. An AI student has completely failed to solve a difficult problem, even after being given a hint. The conceptual leap is too large. 

        **YOUR TASK:**
        Create a new, **simpler problem** that acts as a stepping stone.
        Your new problem should:
        1. REMOVE AT LEAST ONE OF THE CONSTRAINTS FROM THE ORIGINAL PROBLEM.
        2. ENSURE THE SIMPLER PROBLEM IS STRICTLY SMALLER IN LENGTH THAN THE GIVEN HARDER PROBLEM.

        Example to understand how to make a "hard" problem "simpler":

        **Hard Problem**
        "Match lines that start with Order, followed by a space, exactly six digits, a hyphen, three uppercase letters, and nothing else on the line."

        **Simpler Problem**
        "Match lines that start with Order, followed by a space, exactly six digits, and nothing else on the line."

        -----------------------------------------------------------------------

        **Hard Problem**
        “Match lines that start with Receipt, followed by a colon and a space, a three-letter uppercase prefix, exactly four digits, a slash, two lowercase letters, and nothing else on the line.”

        **Simpler Problem**
        "Match lines that start with Receipt: , then exactly four digits, and nothing else on the line."

        -----------------------------------------------------------------------

        **Hard Problem** 
        "Match an entire URL with http or https, optional user:pass@, a valid domain (or IPv4), optional :port, zero or more path segments, optional ? query string, optional # fragment, and nothing else."
        
        **Simpler Problem**: 
        "Match strings that start with http or https and then any sequence of non-space characters." 

        -----------------------------------------------------------------------

        **Hard Problem**
        "Match lines that start with Color:, followed by a space, a #, exactly six uppercase hexadecimal characters (0-9, A-F), and nothing else on the line."

        **Simpler Problem**
        "Match lines that start with Color: # followed by exactly six hexadecimal characters."

        ------------------------------------------------------------------------

        **Hard Problem**
        "Match email addresses where the local part may include letters, digits, `.`, `_`, `%`, `+`, `-` (but not starting or ending with `.`), an `@`, then a domain of letters/digits or hyphens (labels 1–63 chars, no leading/trailing hyphens), a dot, and a TLD of 2–6 letters, with nothing else on the line."

        **Simpler Problem**
        "Match strings that contain non-space characters, then an `@`, then non-space characters, with nothing else on the line."

        ------------------------------------------------------------------------

        **Hard Problem**
        "Match dates in ISO format YYYY‑MM‑DD where YYYY is 1900–2099, MM is 01–12, DD is valid for the given month (including leap-year rules), and nothing else on the line."

        **Simpler Problem**
        "Match strings in the form YYYY‑MM‑DD where each component is the correct number of digits, with nothing else on the line."

        ------------------------------------------------------------------------

        **Hard Problem**
        "Match IPv4 addresses with four octets (0–255) separated by dots, ensuring each octet is within range, and nothing else on the line."

        **Simpler Problem**
        "Match strings that consist of four groups of one to three digits separated by dots, with nothing else on the line."

        ------------------------------------------------------------------------

        **Hard Problem**
        "Match Windows file paths that start with a drive letter (A–Z), colon, backslash, then one or more path segments of letters, digits, spaces, hyphens or underscores (no invalid filename chars), separated by backslashes, and nothing else on the line."

        **Simpler Problem**
        "Match strings that start with a drive letter (A–Z), colon, backslash, then any characters up to the end of the line."

        ------------------------------------------------------------------------

        **The Hard Problem:**
        {hard_problem.description}

        Provide your output as a single JSON object with these exact keys:
            "simplification_strategy", "must_match", "must_not_match", "certification_regex" (the correct regex for your simplified problem/description), "description" (simplified problem), "concepts" (as a list), "level" (as an integer).
        """
        simpler_problem = self._generate_and_certify(prompt)
        return simpler_problem
        
class RegexSolver:
    """
    The agent responsible for solving the problems generated by the ProblemGenerator.
    """
    
    def __init__(self, model_type: str = "huggingface", model_name = "Qwen/Qwen3-4B", **model_kwargs):
        """
        Initialize the RegexSolver with a specific model type.
        
        Args:
            model_type: Either "huggingface" or "gemini"
            **model_kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        if self.model_type=="huggingface":
            self.hf_tokenizer, self.hf_model = load_hf_model(model_name)
        elif self.model_type=="gemini":
            self.gemini_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
    
    def ask_llm(self, prompt: str, model_type: str = "huggingface", **kwargs) -> str:
        """
        Unified interface to ask any LLM model.
        
        Args:
            prompt: The prompt to send
            model_type: Either "huggingface" or "gemini"
            **kwargs: Additional parameters for specific models
        """
        if model_type.lower() == "huggingface":
            return ask_hf_model(prompt=prompt,tokenizer=self.hf_tokenizer,model=self.hf_model, **kwargs)
        elif model_type.lower() == "gemini":
            return ask_gemini(prompt=prompt,llm=self.gemini_llm, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def _ask_model(self, prompt: str) -> str:
        """Internal method to ask the configured model."""
        return self.ask_llm(prompt, self.model_type, **self.model_kwargs)
    
    def solve_problem(self, problem: RegexProblem, relevant_examples_str: str) -> str:
        """
        Attempts to generate a regex solution for a given problem.
        Uses a formatted string of semantically relevant few-shot examples.
        """
        prompt = f"""
        You are a regular expression expert. Your task is to write a single, Python-compatible regex pattern to solve the following problem.
        {relevant_examples_str if relevant_examples_str else ""}
        **New Problem to Solve:**
        - Description: {problem.description}
        - Must Match: {problem.must_match}
        - Must Not Match: {problem.must_not_match}

        Provide only the raw regex pattern as your answer. Do not include any explanation or code fences.
        """
        return self._ask_model(prompt)
    
    def solve_with_hint(self, problem: RegexProblem, hint: str) -> str:
        """
        Attempts to solve a problem using a specific, targeted hint.
        This is used when the system detects the solver is stuck on a new concept.
        """
        prompt = f"""
        You are a regular expression expert. You have failed to solve the following problem on your own.
        A senior engineer has provided a hint to help you.

        **Problem to Solve:**
        - Description: {problem.description}
        - Must Match: {problem.must_match}
        - Must Not Match: {problem.must_not_match}

        **CRUCIAL HINT:** {hint}

        Now, using the hint, please provide a new, corrected regex pattern.
        Provide only the raw regex pattern as your answer.
        """
        return self._ask_model(prompt)

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
        return self._ask_model(prompt)

# --- Usage Examples ---
if __name__ == "__main__":
    # Example 1: Using Hugging Face model
    print("=== Testing with Hugging Face Model ===")
    hf_generator = ProblemGenerator(model_type="huggingface")
    hf_solver = RegexSolver(model_type="huggingface")
    
    # Example 2: Using Gemini model
    print("=== Testing with Gemini Model ===")
    gemini_generator = ProblemGenerator(model_type="gemini")
    gemini_solver = RegexSolver(model_type="gemini")
    
    # Example 3: Mixed approach - generate with one, solve with another
    print("=== Testing Mixed Approach ===")
    problem = hf_generator.generate_initial_problem()
    if problem:
        solution = gemini_solver.solve_problem(problem)
        print(f"Problem: {problem.description}")
        print(f"Solution: {solution}")