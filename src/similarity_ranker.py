# similarity_ranker.py

from typing import List, Optional
from scipy.spatial.distance import cosine
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from data_structures import RegexSolution, ArchiveCell

class SimilarityRanker:
    """
    A class to handle semantic similarity for finding the best few-shot examples.

    This class maintains a "knowledge base" of previously solved problems and their
    text embeddings. When given a new problem, it can find the 'k-nearest neighbors'
    (k-NN) from its knowledge base to provide highly relevant examples for in-context
    learning.
    """
    def __init__(self):
        print("Initializing semantic similarity ranker with Gemini embeddings...")
        try:
            # Using the specific model you requested
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") # Updated model name
            # This list will store ArchiveCells, which conveniently hold both the
            # problem and its pre-computed embedding vector.
            self.knowledge_base: List[ArchiveCell] = []
            print("Embedding model loaded.")
        except Exception as e:
            print(f"FATAL: Could not initialize GoogleGenerativeAIEmbeddings. Is GOOGLE_API_KEY set? Error: {e}")
            self.embedding_model = None

    def add_to_knowledge_base(self, solution: RegexSolution):
        """
        Embeds a problem's description and adds it to the knowledge base.
        This should only be called for successfully solved problems.
        """
        if not self.embedding_model or not solution.succeeded:
            return

        print(f"Embedding and adding problem to knowledge base: '{solution.problem.description[:50]}...'")
        
        # Embed the problem description and store the vector
        problem_embedding = self.embedding_model.embed_query(solution.problem.description)
        
        # Create a new ArchiveCell to store in our knowledge base
        new_cell = ArchiveCell(
            problem=solution.problem,
            solution_regex=solution.proposed_regex,
            embedding=problem_embedding # Store the vector
        )
        self.knowledge_base.append(new_cell)

    def get_top_k_examples(self, current_problem: 'RegexProblem', k: int) -> str:
        """
        Finds the k most semantically similar problems from the knowledge base.
        """
        if not self.embedding_model or not self.knowledge_base:
            return ""

        # 1. Embed the new, unsolved problem
        current_embedding = self.embedding_model.embed_query(current_problem.description)

        # 2. Calculate cosine distance to all problems in the knowledge base
        distances = []
        for cell in self.knowledge_base:
            if cell.embedding:
                dist = cosine(current_embedding, cell.embedding)
                distances.append((dist, cell))

        # 3. Sort by distance (smallest distance = most similar)
        distances.sort(key=lambda x: x[0])

        # 4. Get the top k examples and format them for the prompt
        top_k = distances[:k]
        if not top_k:
            return ""

        # 5. Format the examples into a string for the prompt
        example_str = "Here are the most relevant case studies from past successes:\n\n"
        for i, (dist, cell) in enumerate(top_k):
            example_str += f"--- Case Study {i+1} (Similarity Score: {1-dist:.2f}) ---\n"
            example_str += f"Problem: {cell.problem.description}\n"
            example_str += f"Solution Regex: `{cell.solution_regex}`\n\n"
        
        return example_str