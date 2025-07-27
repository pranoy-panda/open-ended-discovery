# main.py

import csv
import json
import random
from data_structures import RegexProblem, RegexSolution, QualityDiversityArchive
from harness import evaluate_solution
from llm_agents import ProblemGenerator, RegexSolver

# --- Configuration ---
K_ROUNDS = 20 # Number of discovery rounds to run
MAX_DEBUG_ATTEMPTS = 2
OUTPUT_LOG_FILE = "outputs/evolution_log.csv"
OUTPUT_SUITE_FILE = "outputs/problem_suite_for_gemini.json"

def main():
    """
    Main execution loop for the QD-POET Regex Scientist algorithm.
    """
    print("--- Starting QD-POET Regex Scientist ---")

    # 1. Initialization
    generator = ProblemGenerator()
    solver = RegexSolver()
    archive = QualityDiversityArchive()
    evolution_log = []

    # 2. Kickstart the process with a seed problem
    print("Round 0: Generating seed problem...")
    current_problem = generator.generate_initial_problem()
    if not current_problem:
        print("Fatal: Could not generate a seed problem. Exiting.")
        return

    # 3. Main Open-Ended Discovery Loop
    for k in range(1, K_ROUNDS + 1):
        print(f"\n--- Round {k}/{K_ROUNDS} ---")

        # A. SOLVE the current problem
        solution = None
        for attempt in range(MAX_DEBUG_ATTEMPTS):
            # Use examples from the archive for in-context learning
            few_shot_examples = [
                {"problem_desc": cell.problem.description, "regex": cell.solution_regex}
                for cell in archive.archive.values() if cell.problem
            ]
            
            if attempt == 0:
                print(f"Solving: '{current_problem.description}'")
                proposed_regex = solver.solve_problem(current_problem, random.sample(few_shot_examples, min(len(few_shot_examples), 2)))
            else: # Debugging attempt
                print(f"DEBUG (Attempt {attempt}): Fixing previous solution.")
                proposed_regex = solver.debug_problem(current_problem, solution.proposed_regex, solution.failed_on)

            if not proposed_regex:
                print("Solver failed to provide a response.")
                continue

            solution = RegexSolution(problem=current_problem, proposed_regex=proposed_regex)
            solution = evaluate_solution(solution)

            if solution.is_correct:
                print(f"SUCCESS! Regex: {solution.proposed_regex}")
                solution.succeeded = True
                break
            else:
                print(f"FAILURE. Errors: {solution.failed_on}")
        
        # B. ARCHIVE the successful solution
        if solution and solution.succeeded:
            # Log for visualization
            evolution_log.append({
                "round": k,
                "level": solution.problem.level,
                "concepts": ", ".join(solution.problem.concepts),
                "problem_description": solution.problem.description,
                "successful_regex": solution.proposed_regex
            })
            # Add to the QD archive
            archive.add_to_archive(solution, k)

        # C. EVOLVE to create the next problem (POET step)
        print("Evolving to a new problem...")
        solved_problems = archive.get_all_solved_problems()
        if not solved_problems:
             # If we somehow failed the first problem, try again with the same one
             print("Sticking with current problem as no successes yet.")
             continue
        
        # Select a random solved problem to mutate
        problem_to_mutate = random.choice(solved_problems)
        new_problem = generator.mutate_problem(problem_to_mutate)
        
        if new_problem:
            current_problem = new_problem
        else:
            print("WARN: Failed to mutate. Re-using last successful problem for mutation.")
            current_problem = problem_to_mutate # Fallback

    # 4. Finalization: Save all outputs
    print("\n--- Discovery Complete. Saving outputs. ---")

    # Save the evolution log
    with open(OUTPUT_LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "level", "concepts", "problem_description", "successful_regex"])
        writer.writeheader()
        writer.writerows(evolution_log)
    print(f"Evolution log saved to {OUTPUT_LOG_FILE}")

    # Create and save the problem suite for Gemini
    problem_suite = {"easy": None, "medium": None, "hard": None}
    for cell in archive.archive.values():
        coords = archive.get_coordinates(cell.problem)
        problem_dict = {
            "description": cell.problem.description,
            "must_match": cell.problem.must_match,
            "must_not_match": cell.problem.must_not_match
        }
        if coords == "basic" and not problem_suite["easy"]:
            problem_suite["easy"] = problem_dict
        elif coords in ["grouping", "quantifier_range"] and not problem_suite["medium"]:
            problem_suite["medium"] = problem_dict
        elif coords == "lookaround" and not problem_suite["hard"]:
            problem_suite["hard"] = problem_dict

    with open(OUTPUT_SUITE_FILE, "w") as f:
        json.dump(problem_suite, f, indent=4)
    print(f"Problem suite for testing saved to {OUTPUT_SUITE_FILE}")


if __name__ == "__main__":
    main()