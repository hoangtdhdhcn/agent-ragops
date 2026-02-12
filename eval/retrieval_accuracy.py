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
    Measure how relevant the retrieved documents are to the user question.

    Design rule:
    - This module ONLY calls the LLM
    - It returns RAW JSON text
    - Parsing and validation are handled downstream (Pydantic)
"""

# -------------------------------
# Retrieval relevance prompt (STRICT JSON)
# -------------------------------
RETRIEVAL_RELEVANCE_GRADER_PROMPT = """
    You are an evaluator for an information retrieval system.

    You will be given:
    - A QUESTION
    - FACTS (retrieved documents)

    Task:
    Determine whether the FACTS are relevant to the QUESTION.

    Rules:
    1. Judge relevance ONLY.
    2. If the facts contain ANY keywords, entities, or semantic meaning related to the question, they ARE relevant.
    3. It is acceptable if parts of the facts are unrelated.
    4. If the facts are completely unrelated, they are NOT relevant.
    5. Do NOT require completeness or correctness.

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

    FACTS:
    {facts}
"""

# -------------------------------
# Gemini grader LLM
# -------------------------------
retrieval_relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# -------------------------------
# Retrieval relevance evaluator
# -------------------------------
def retrieval_relevance(inputs: dict, outputs: dict):
    """
    Retrieval relevance evaluator.

    Returns:
        str: RAW LLM output (JSON string)

    NOTE:
    - No JSON parsing happens here
    - Fail-closed behavior returns valid JSON
    """

    question = inputs.get("question", "")
    documents = outputs.get("documents", [])

    facts = "\n\n".join(
        getattr(doc, "page_content", "") for doc in documents
    )

    prompt = RETRIEVAL_RELEVANCE_GRADER_PROMPT.format(
        question=question,
        facts=facts,
    )

    try:
        response = retrieval_relevance_llm.invoke(
            [
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ]
        )

        # 🔑 Return RAW text only
        return response.content

    except Exception as e:
        logger.error(f"Retrieval relevance grader failed: {e}")
        # Fail closed with valid JSON
        return '{"explanation": "LLM invocation failed", "result": false}'
