"""LLM-as-Judge faithfulness evaluator for RAG answers."""

import json

from loguru import logger

from src.providers.factory import get_llm_provider

FAITHFULNESS_PROMPT = """\
You are an expert answer-quality evaluator for a document Q&A system.

Given the QUESTION, the retrieved CONTEXT passages, and the generated ANSWER,
evaluate how **faithful** and **grounded** the answer is in the provided context.

<question>
{question}
</question>

<context>
{context}
</context>

<answer>
{answer}
</answer>

Scoring rubric:
- 1.0: Answer is fully and accurately supported by the context.
- 0.8: Answer is mostly supported with minor extrapolations that are reasonable.
- 0.6: Answer is partially supported but contains some unsupported claims.
- 0.4: Answer has weak support; significant portions are not grounded.
- 0.2: Answer is mostly unsupported or speculative.
- 0.0: Answer contradicts the context or is entirely fabricated.

Consider:
1. Does the answer make claims not present in the context?
2. Does the answer accurately represent numbers, dates, and facts from the context?
3. Does the answer contradict any information in the context?
4. For calculations, are the steps logically sound given the source data?

Respond with ONLY valid JSON (no markdown fences):
{{"score": <float>, "reason": "<one sentence explanation>"}}"""


async def compute_confidence(result: dict, query: str) -> tuple[float, str]:
    """Compute answer confidence via LLM faithfulness evaluation.

    Returns:
        Tuple of (score: float 0-1, reason: str).
    """
    citations = result.get("citations", [])
    answer = result.get("answer", "")

    if not answer.strip():
        return 0.0, "Empty answer"

    if not citations:
        return 0.0, "No citations retrieved"

    # Build context block from citations
    parts = []
    for i, c in enumerate(citations, 1):
        source = c.get("source", "document_text")
        table = c.get("table_name")
        header = f"[{i}] Page {c.get('page', '?')} | {source}"
        if table:
            header += f" | Table: {table}"
        parts.append(f"{header}\n{c.get('text', '')}")
    context_block = "\n\n".join(parts)

    prompt = FAITHFULNESS_PROMPT.format(
        question=query,
        context=context_block,
        answer=answer,
    )

    llm = get_llm_provider()
    raw = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise evaluation assistant. "
                    'Respond with ONLY a JSON object in this exact format: {"score": <float 0-1>, "reason": "<one sentence>"}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    # Strip markdown fences if the model wraps them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    evaluation = json.loads(cleaned)
    score = max(0.0, min(1.0, float(evaluation["score"])))
    reason = evaluation.get("reason", "")

    logger.info(f"Faithfulness evaluation: score={score}, reason={reason}")
    return score, reason
