from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import os

logger = logging.getLogger(__name__)

# -------------------------------
# Gemini API key
# -------------------------------
os.environ.get("GEMINI_API_KEY")

"""
    Goal:
    Measure how well the generated answer addresses the user's question.

    Design rule:
    - This module ONLY calls the LLM
    - It returns RAW JSON text
    - Parsing and validation are handled elsewhere (Pydantic)
"""

# -------------------------------
# Relevance prompt (STRICT JSON)
# -------------------------------
RELEVANCE_GRADER_PROMPT = """
    You are an evaluator for a question-answering system.

    You will be given:
    - A QUESTION
    - A MODEL ANSWER

    Task:
    Determine whether the MODEL ANSWER is relevant and helpful for answering the QUESTION.

    Rules:
    1. Judge relevance ONLY (not factual correctness).
    2. The answer must directly address the intent of the question.
    3. Off-topic, generic, or evasive answers are NOT relevant.
    4. Verbosity is acceptable only if the answer still clearly addresses the question.
    5. If the answer does not meaningfully help, it is NOT relevant.

    Output rules (MANDATORY):
    - Return ONLY valid JSON
    - Do NOT use markdown
    - Do NOT use triple backticks
    - Do NOT include any text outside JSON

    Required JSON format:
    {
    "explanation": "brief reasoning",
    "result": true or false
    }

    QUESTION:
    {question}

    MODEL ANSWER:
    {answer}
"""

# -------------------------------
# Gemini grader LLM
# -------------------------------
relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# -------------------------------
# Relevance evaluator
# -------------------------------
def relevance(inputs: dict, outputs: dict):
    """
    Relevance evaluator for RAG systems.

    Returns:
        str: RAW LLM output (JSON string)

    NOTE:
    - No JSON parsing here
    - Fail-closed behavior returns valid JSON
    """

    question = inputs.get("question", "")
    answer = outputs.get("answer", "")

    prompt = RELEVANCE_GRADER_PROMPT.format(
        question=question,
        answer=answer,
    )

    try:
        response = relevance_llm.invoke(
            [
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ]
        )

        # Return RAW text only
        return response.content

    except Exception as e:
        logger.error(f"Relevance grader failed: {e}")
        # Fail closed with valid JSON
        return '{"explanation": "LLM invocation failed", "result": false}'
