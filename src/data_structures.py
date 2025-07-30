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
    embedding: Optional[List[float]] = None

class QualityDiversityArchive:
    """
    A grid to store a collection of problems that are both high-quality (solved) and
    diverse. This new version allows for improvement by replacing problems in a niche
    if a more complex (higher level) one is solved.
    """
    def __init__(self):
        self.archive: Dict[str, ArchiveCell] = {}
        self.STATUS_IMPROVED_NICHE = "IMPROVED_NICHE" # New status for replacement
        self.STATUS_FILLED_NEW_NICHE = "FILLED_NEW_NICHE"
        self.STATUS_NOT_ADDED = "NOT_ADDED"

    def get_coordinates(self, problem: RegexProblem) -> str:
        """Determines the 'coordinates' of a problem based on its concepts."""
        if "backreference" in problem.concepts: return "backreference"
        if "lookaround" in problem.concepts: return "lookaround"
        if "quantifier_range" in problem.concepts: return "quantifier_range"
        if "grouping" in problem.concepts: return "grouping"
        return "basic"

    def add_to_archive(self, solution: RegexSolution, round_num: int) -> str:
        """
        Adds a successfully solved problem to the archive. It will now replace an
        existing problem in a niche if the new one has a higher level.
        """
        if not solution.succeeded:
            return self.STATUS_NOT_ADDED

        coords = self.get_coordinates(solution.problem)

        # Check if the niche is empty OR if the new problem is better.
        if coords not in self.archive or self.archive[coords].problem is None or self.archive[coords].problem.level < solution.problem.level:
            is_improvement = coords in self.archive and self.archive[coords].problem is not None
            
            self.archive[coords] = ArchiveCell(
                problem=solution.problem,
                solution_regex=solution.proposed_regex,
                solved_at_round=round_num
            )
            # Return a different status if we replaced an old solution.
            return self.STATUS_IMPROVED_NICHE if is_improvement else self.STATUS_FILLED_NEW_NICHE
            
        return self.STATUS_NOT_ADDED

    def get_all_solved_problems(self) -> List[RegexProblem]:
        """Returns all unique problems stored in the archive."""
        return [cell.problem for cell in self.archive.values() if cell.problem]