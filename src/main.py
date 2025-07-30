# open-ended-discovery/src/main.py
import os
import csv
import json
import random
from data_structures import RegexProblem, RegexSolution, QualityDiversityArchive
from harness import evaluate_solution, analyze_regex_concepts
from llm_agents import ProblemGenerator, RegexSolver
from similarity_ranker import SimilarityRanker 

# --- Configuration ---
K_ROUNDS = 10 # Number of discovery rounds to run
MAX_DEBUG_ATTEMPTS = 1
ROAD_BLOCK_THRESHOLD = 2
SOLVER_MODEL_TYPE = "gemini"
SOLVER_MODEL_NAME = "gemini-2.5-flash"
GENERATOR_MODEL_TYPE = "gemini"
GENERATOR_MODEL_NAME = "gemini-2.0-flash"
OUTPUT_LOG_FILE = "outputs/evolution_log.csv"
OUTPUT_SUITE_FILE = "outputs/problem_suite_for_gemini.json"

def main():
    """
    Main execution loop for the QD-POET Regex Scientist algorithm.
    """
    print("--- Starting QD-POET Regex Scientist ---")

    # 1. Initialization
    generator = ProblemGenerator(model_type=GENERATOR_MODEL_TYPE, model_name=GENERATOR_MODEL_NAME)
    solver = RegexSolver(model_type=SOLVER_MODEL_TYPE, model_name=SOLVER_MODEL_NAME)
    archive = QualityDiversityArchive()
    ranker = SimilarityRanker()
    evolution_log = []
    recent_problem_descriptions = []
    failed_problems_history = []
    roadblock_counter = 0

    # 2. Kickstart the process with a seed problem
    print("\n--- Round 0: Seeding Process ---")
    current_problem, current_generator_regex = generator.generate_initial_problem()
    if not current_problem:
        print("Fatal: Could not generate a seed problem. Exiting.")
        return
    
    print(f"GENERATOR: Created seed problem (Level {current_problem.level}): '{current_problem.description}'")
    recent_problem_descriptions.append(current_problem.description) # Seed the history

    # 3. Main Open-Ended Discovery Loop
    for k in range(1, K_ROUNDS + 1):
        print(f"\n--- Round {k}/{K_ROUNDS} ---")

        # A. SOLVE the current problem
        solution = None
        is_solved = False

        # Check if we are stuck and need to provide help
        if roadblock_counter >= ROAD_BLOCK_THRESHOLD: # Trigger help after 2 failed rounds 
            roadblock_counter = 0
            print("GENERATOR: Attempting to generate a simpler stepping-stone problem.")
            simplified_problem, simplified_gen_regex = generator.simplify_problem(
                current_problem, failed_problems_history[-1].failed_on
            )

            if simplified_problem:
                print(f"GENERATOR: Created simplified problem (Level {simplified_problem.level}): '{simplified_problem.description}'")
                print("SOLVER: Attempting the new simplified problem immediately...")
                
                relevant_examples = ranker.get_top_k_examples(simplified_problem, k=2)
                simple_regex = solver.solve_problem(simplified_problem, relevant_examples)
                simple_solution = RegexSolution(problem=simplified_problem, proposed_regex=simple_regex)
                simple_solution = evaluate_solution(simple_solution)

                if simple_solution.is_correct:
                    print("SOLVER: SUCCESS on the simplified problem! This will be added to the archive.")
                    # Overwrite the main solution variables with this partial success.
                    # This injects the stepping-stone into the system's history for the round.
                    solution = simple_solution
                    is_solved = True
                    # keep the generator regex for the problem that was actually solved
                    current_generator_regex = simplified_gen_regex
                    roadblock_counter = 0 # made progress, so reset the counter.
                else:
                    print("SOLVER: Failed even on the simplified problem. This round is a final failure.")

        if not is_solved:
            for attempt in range(MAX_DEBUG_ATTEMPTS):
                if attempt == 0:
                    print(f"Solving: '{current_problem.description}'")
                    relevant_examples = ranker.get_top_k_examples(current_problem, k=2)
                    proposed_regex = solver.solve_problem(current_problem, relevant_examples)
                else:
                    print(f"DEBUG (Attempt {attempt}): Fixing previous solution.")
                    proposed_regex = solver.debug_problem(current_problem, solution.proposed_regex, solution.failed_on)

                if not proposed_regex:
                    print("Solver failed to provide a response.")
                    continue

                solution = RegexSolution(problem=current_problem, proposed_regex=proposed_regex)
                solution = evaluate_solution(solution)

                if solution and solution.is_correct:
                    print(f"SOLVER: SUCCESS! Regex: {solution.proposed_regex}")
                    solution.succeeded = True
                    is_solved = True
                    break
                else:
                    print(f"SOLVER: FAILURE. Errors: {solution.failed_on if solution else 'No response'}")

        # B. UPDATE HISTORY based on the outcome of this round
        if is_solved and solution:
            # Analyze the solution to get verified concepts
            verified_concepts = analyze_regex_concepts(solution.proposed_regex)
            print(f"SYSTEM: Verified concepts in solution: {verified_concepts}")
            solution.problem.concepts = verified_concepts # Update the problem with the ground truth

            # Now, add the correctly labeled solution to the archive and ranker
            status = archive.add_to_archive(solution, k)
    
            # Only add to ranker if it's a new or improved problem to keep examples fresh
            if status != archive.STATUS_NOT_ADDED:
                ranker.add_to_knowledge_base(solution)

            if status == archive.STATUS_IMPROVED_NICHE:
                print(f"QD: Progress! Replaced a simpler problem in niche '{archive.get_coordinates(solution.problem)}' with this more advanced one.")
            elif status == archive.STATUS_FILLED_NEW_NICHE:
                print(f"QD: Groundbreaking! Discovered and solved a new problem type: '{archive.get_coordinates(solution.problem)}'")

            evolution_log.append({
                "round": k,
                "level": solution.problem.level,
                "concepts": ", ".join(solution.problem.concepts),
                "problem_description": solution.problem.description,
                "generator_certified_regex": current_generator_regex, # The regex from the problem's creator
                "successful_solver_regex": solution.proposed_regex, # The regex from the separate solver agent
                "must_match_cases": json.dumps(solution.problem.must_match),
                "must_not_match_cases": json.dumps(solution.problem.must_not_match)
            })
        elif solution: # some regex generated but not correct
            print(f"SOLVER: FINAL FAILURE for problem: '{current_problem.description}'")
            failed_problems_history.append(solution)

            evolution_log.append({
                "round": k,
                "level": solution.problem.level,
                "concepts": ", ".join(solution.problem.concepts),
                "problem_description": solution.problem.description,
                "generator_certified_regex": current_generator_regex, # The regex from the problem's creator
                "successful_solver_regex": "FAILURE", # The regex from the separate solver agent
                "must_match_cases": json.dumps(solution.problem.must_match),
                "must_not_match_cases": json.dumps(solution.problem.must_not_match)
            })


        # C. EVOLVE to create the NEXT problem for the next round
        successful_solutions = archive.get_all_solved_problems()
        if not successful_solutions:
            roadblock_counter += 1
            print("EVOLVE: No successful problems in archive yet. Re-attempting current problem.")
            continue

        problem_to_mutate = random.choice(successful_solutions)
        print(f"POET: Selected problem (Level {problem_to_mutate.level}) to evolve: '{problem_to_mutate.description}'")
        
        num_success = len(successful_solutions)
        num_failure = len(failed_problems_history)
        print(f"GENERATOR: Analyzing {num_success} successes and {num_failure} failures to inform creation.")
        new_problem, new_generator_regex = generator.generate_next_problem(
            successful_solutions, failed_problems_history, recent_problem_descriptions
        )

        if new_problem:
            current_problem = new_problem
            current_generator_regex = new_generator_regex

            print(f"GENERATOR: Created new problem (Level {current_problem.level}): '{current_problem.description}'")
            recent_problem_descriptions.append(current_problem.description)
            if len(recent_problem_descriptions) > 4: # Keep the list size manageable
                recent_problem_descriptions.pop(0)
        else:
            print("WARN: Generator failed to create a valid new problem. Re-using mutation base for next round.")
            current_problem = problem_to_mutate
            current_generator_regex = "N/A (re-used problem)"

    # 4. Finalization: Save all outputs
    print("\n--- Discovery Complete. Saving outputs. ---")

    # Save the evolution log
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)

    # Update fieldnames to match the new rich data
    with open(OUTPUT_LOG_FILE, "w", newline="", encoding='utf-8') as f:
        fieldnames = [
            "round", "level", "concepts", "problem_description", 
            "generator_certified_regex", "successful_solver_regex",
            "must_match_cases", "must_not_match_cases"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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