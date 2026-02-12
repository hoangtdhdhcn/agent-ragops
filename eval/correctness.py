from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import logging
import os

# Set Gemini API key here
os.getenv("GEMINI_API_KEY")

""" Goal: Measure “how similar/correct is the RAG chain answer, relative to a ground-truth answer”
    Mode: Requires a ground truth (reference) answer supplied through a dataset
    Evaluator: Use LLM-as-judge to assess answer correctness.
"""

logger = logging.getLogger(__name__)

# -------------------------------
# Output schema (for validation)
# -------------------------------
class CorrectnessGrade(TypedDict):
    explanation: str
    correct: bool

# -------------------------------
# Correctness prompt
# -------------------------------
correctness_instructions = """
    You are evaluating the correctness of a model-generated answer.

    You will be given:
    - A QUESTION
    - A REFERENCE ANSWER (ground truth)
    - A MODEL ANSWER

    Task:
    Determine whether the MODEL ANSWER is factually correct relative to the REFERENCE ANSWER.

    Rules:
    1. Judge factual accuracy ONLY using the reference answer.
    2. The model answer may paraphrase the reference answer.
    3. Additional information is allowed only if it does not contradict the reference.
    4. If any contradiction exists, the answer is incorrect.
    5. If the model answer omits required facts, is ambiguous, or relies on unstated assumptions, the answer is incorrect.
    6. Do not infer missing facts.

    Return ONLY valid JSON (can't contain ```json ```) in the following format:
    {
    "explanation": "step-by-step reasoning",
    "result": true or false
    }
"""

# -------------------------------
# Gemini grader LLM
# -------------------------------
grader_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# -------------------------------
# Correctness evaluator
# -------------------------------
def correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    """
    Correctness evaluator for RAG systems.

    Evaluates:
        Model Answer vs Reference Answer

    Returns:
        bool: True if correct, False otherwise
    """

    prompt = f"""
    QUESTION:
    {inputs.get("question", "")}

    REFERENCE ANSWER:
    {reference_outputs.get("answer", "")}

    MODEL ANSWER:
    {outputs.get("answer", "")}
"""

    try:
        response = grader_llm.invoke(
            [
                {"role": "system", "content": correctness_instructions},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        logger.error(f"Gemini invocation failed: {e}")
        return False  # fail closed

    # Gemini returns text → must parse JSON
    try:
        grade = json.loads(response.content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from grader: {response.content}")
        return False  # fail closed

    # Validate schema strictly
    if not isinstance(grade, dict):
        return False

    explanation = grade.get("explanation")
    correct = grade.get("correct")

    if not isinstance(explanation, str):
        return False
    if not isinstance(correct, bool):
        return False

    return correct
