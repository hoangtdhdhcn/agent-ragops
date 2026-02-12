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
    Measure to what extent the generated answer is grounded in the retrieved context.

    Important design rule:
    - This module ONLY calls the LLM and returns RAW JSON text
    - Parsing and validation are handled elsewhere (Pydantic)
"""

# -------------------------------
# Groundedness prompt (STRICT JSON)
# -------------------------------
GROUNDING_GRADER_PROMPT = """
    You are an evaluator for a RAG system.

    You will be given:
    - FACTS (retrieved documents)
    - MODEL ANSWER

    Task:
    Determine whether the MODEL ANSWER is fully grounded in the FACTS.

    Rules:
    1. Judge grounding ONLY using the provided facts.
    2. Every factual claim must be supported by the facts.
    3. Paraphrasing is allowed.
    4. If the answer introduces information not in the facts, it is NOT grounded.
    5. If facts are insufficient, it is NOT grounded.
    6. Do NOT use external knowledge.

    Output rules (MANDATORY):
    - Return ONLY valid JSON
    - Do NOT use markdown
    - Do NOT use triple backticks
    - Do NOT include any text outside the JSON

    Required JSON format:
    {
    "explanation": "brief reasoning",
    "result": true or false
    }

    FACTS:
    {facts}

    MODEL ANSWER:
    {answer}
"""

# -------------------------------
# Gemini grader LLM
# -------------------------------
grounded_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# -------------------------------
# Groundedness evaluator
# -------------------------------
def groundedness(inputs: dict, outputs: dict):
    """
    Groundedness evaluator for RAG systems.

    Returns:
        str: RAW LLM output (JSON string)

    NOTE:
    - No JSON parsing happens here
    - Fail-closed behavior returns valid JSON
    """

    documents = outputs.get("documents", [])
    facts = "\n\n".join(
        getattr(doc, "page_content", "") for doc in documents
    )

    answer = outputs.get("answer", "")

    prompt = GROUNDING_GRADER_PROMPT.format(
        facts=facts,
        answer=answer,
    )

    try:
        response = grounded_llm.invoke(
            [
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ]
        )

        # Return RAW text only
        return response.content

    except Exception as e:
        logger.error(f"Groundedness grader failed: {e}")
        # Fail closed with valid JSON
        return '{"explanation": "LLM invocation failed", "result": false}'
