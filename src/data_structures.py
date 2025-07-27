# data_structures.py
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- Core Data Objects ---

@dataclass
class RegexProblem:
    """Represents a single, well-defined regex challenge."""
    description: str
    must_match: List[str]
    must_not_match: List[str]
    # Metadata for the QD Archive
    concepts: List[str] = field(default_factory=list)
    level: int = 1

@dataclass
class RegexSolution:
    """Represents a proposed solution and its evaluation status."""
    problem: RegexProblem
    proposed_regex: str
    is_correct: bool = False
    failed_on: List[str] = field(default_factory=list)
    succeeded: bool = False # Final success status after potential debugging

# --- Quality-Diversity Component ---

@dataclass
class ArchiveCell:
    """A single cell in our QD archive, holding the best problem found for a niche."""
    problem: Optional[RegexProblem] = None
    solution_regex: Optional[str] = None
    solved_at_round: Optional[int] = None

class QualityDiversityArchive:
    """
    A grid to store a collection of problems that are both high-quality (solved) and
    diverse (cover different types of challenges). This is the core of the QD approach,
    preventing the system from only pursuing one type of difficulty.
    """
    def __init__(self):
        # The archive is a dictionary where keys are diversity characteristics.
        # For our task, the key is a string representing the primary regex concept.
        self.archive: Dict[str, ArchiveCell] = {}

    def get_coordinates(self, problem: RegexProblem) -> str:
        """
        Determines the "coordinates" of a problem in the archive based on its
        most advanced concept.
        """
        # Simple mapping for demonstration. This can be made more sophisticated.
        if "lookaround" in problem.concepts:
            return "lookaround"
        if "quantifier_range" in problem.concepts:
            return "quantifier_range"
        if "grouping" in problem.concepts:
            return "grouping"
        return "basic"

    def add_to_archive(self, solution: RegexSolution, round_num: int) -> bool:
        """
        Adds a successfully solved problem to the archive.

        It only adds the solution if the corresponding cell is empty, thus preserving
        the first discovered solution for that niche. Returns True if added.
        """
        if not solution.succeeded:
            return False

        coords = self.get_coordinates(solution.problem)
        if coords not in self.archive or self.archive[coords].problem is None:
            self.archive[coords] = ArchiveCell(
                problem=solution.problem,
                solution_regex=solution.proposed_regex,
                solved_at_round=round_num
            )
            print(f"ARCHIVE: Added new problem to niche '{coords}'.")
            return True
        return False

    def get_all_solved_problems(self) -> List[RegexProblem]:
        """Returns all unique problems stored in the archive."""
        return [cell.problem for cell in self.archive.values() if cell.problem]