import json
import logging
from typing import Any

from app.features.analyze.db import analysis
from app.features.analyze.model import analyze_model
from app.features.analyze.schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeSegmentResult,
)


logger = logging.getLogger(__name__)


async def analyze_transcription(
    request: AnalyzeRequest,
) -> AnalyzeResponse:

    logger.info(
        "Starting transcript analysis | "
        "job_id=%s | video_id=%s | segments=%d",
        request.job_id,
        request.video_id,
        len(request.segments),
    )

    if not request.segments:

        logger.warning(
            "No segments received for analysis | "
            "job_id=%s | video_id=%s",
            request.job_id,
            request.video_id,
        )

        empty_response = AnalyzeResponse(
            job_id=request.job_id,
            video_id=request.video_id,
            segments=[],
        )

        return empty_response

    # ---------------------------------------------------------
    # PREPARE SEGMENT TEXT
    # ---------------------------------------------------------

    texts = [
        segment.text.strip()
        for segment in request.segments
    ]

    logger.info(
        "Generating embeddings | "
        "job_id=%s | texts=%d",
        request.job_id,
        len(texts),
    )

    # ---------------------------------------------------------
    # GENERATE EMBEDDINGS
    # ---------------------------------------------------------

    embeddings = analyze_model.embed_model.encode(
        texts,
        convert_to_numpy=True,
    )

    logger.info(
        "Embeddings generated | "
        "job_id=%s | embeddings=%d",
        request.job_id,
        len(embeddings),
    )

    results: list[AnalyzeSegmentResult] = []

    # ---------------------------------------------------------
    # ANALYZE EACH SEGMENT
    # ---------------------------------------------------------

    for index, segment in enumerate(request.segments):

        logger.info(
            "Analyzing segment | "
            "job_id=%s | segment_index=%d | text=%r",
            request.job_id,
            segment.index,
            segment.text,
        )

        scores = await score_segment_with_llm(
            text=segment.text,
            language=request.language,
        )

        results.append(
            AnalyzeSegmentResult(
                index=segment.index,
                embedding=embeddings[index].tolist(),
                semantic_score=scores["semantic_score"],
                novelty_score=scores["novelty_score"],
                clarity_score=scores["clarity_score"],
                hook_score=scores["hook_score"],
                semantic_labels=scores["semantic_labels"],
                suggested_hook=scores["suggested_hook"],
            )
        )

    logger.info(
        "Transcript analysis completed | "
        "job_id=%s | video_id=%s | results=%d",
        request.job_id,
        request.video_id,
        len(results),
    )

    # ---------------------------------------------------------
    # SAVE ANALYSIS TO MONGODB
    # ---------------------------------------------------------

    logger.info(
        "Saving transcript analysis to MongoDB | "
        "job_id=%s | video_id=%s",
        request.job_id,
        request.video_id,
    )

    analysis_segments = [
        result.model_dump()
        for result in results
    ]

    analysis_id = analysis.save_analysis(
        job_id=request.job_id,
        video_id=request.video_id,
        user_id=request.user_id,
        language=request.language,
        segments=analysis_segments,
    )

    logger.info(
        "Transcript analysis saved successfully | "
        "job_id=%s | video_id=%s | analysis_id=%s",
        request.job_id,
        request.video_id,
        analysis_id,
    )

    # ---------------------------------------------------------
    # RETURN COMPLETE ANALYSIS RESPONSE
    # ---------------------------------------------------------

    response = AnalyzeResponse(
        job_id=request.job_id,
        video_id=request.video_id,
        segments=results,
    )

    logger.info(
        "Final analysis response prepared | "
        "job_id=%s | video_id=%s | segments=%d",
        request.job_id,
        request.video_id,
        len(response.segments),
    )

    return response


async def score_segment_with_llm(
    text: str,
    language: str,
) -> dict[str, Any]:

    if analyze_model.groq_client is None:

        logger.warning(
            "Groq client unavailable | "
            "Returning default analysis scores"
        )

        return {
            "semantic_score": 0.0,
            "novelty_score": 0.0,
            "clarity_score": 0.0,
            "hook_score": 0.0,
            "semantic_labels": [],
            "suggested_hook": "",
        }

    prompt = f"""
Analyze the following transcript segment for short-form
video potential.

Evaluate the segment based on:

1. semantic_score
   How meaningful and information-rich is the content?

2. novelty_score
   How unusual, fresh, surprising, or distinctive is the content?

3. clarity_score
   How clear and understandable is the segment?

4. hook_score
   How strong is the segment as a hook for a short video?

5. semantic_labels
   Provide concise topic or meaning labels.

6. suggested_hook
   Write one short hook based ONLY on the segment.
   Do not invent facts.

Rules:

- Return ONLY valid JSON.
- Scores must be between 0.0 and 1.0.
- Analyze the actual meaning and context of the segment.
- Do not translate the segment.
- Support Hindi, English, Hinglish, and mixed-language speech.
- Do not invent information that is not present in the segment.
- Keep semantic_labels concise.
- suggested_hook should be short and attention-grabbing.

Language:
{language}

Segment:
{text[:1000]}

Return exactly:

{{
    "semantic_score": 0.0,
    "novelty_score": 0.0,
    "clarity_score": 0.0,
    "hook_score": 0.0,
    "semantic_labels": [],
    "suggested_hook": ""
}}
"""

    try:

        response = analyze_model.groq_client.chat.completions.create(
            model=analyze_model.GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a multilingual short-form video "
                        "content analysis system. "
                        "Return ONLY valid JSON. "
                        "Never return markdown or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_completion_tokens=512,
            response_format={
                "type": "json_object",
            },
        )

        content = response.choices[0].message.content

        if not content:

            raise ValueError(
                "Groq returned an empty response"
            )

        result = json.loads(content)

        # -----------------------------------------------------
        # VALIDATE / NORMALIZE SCORES
        # -----------------------------------------------------

        semantic_score = _normalize_score(
            result.get("semantic_score", 0.0)
        )

        novelty_score = _normalize_score(
            result.get("novelty_score", 0.0)
        )

        clarity_score = _normalize_score(
            result.get("clarity_score", 0.0)
        )

        hook_score = _normalize_score(
            result.get("hook_score", 0.0)
        )

        semantic_labels = result.get(
            "semantic_labels",
            [],
        )

        if not isinstance(
            semantic_labels,
            list,
        ):
            semantic_labels = []

        semantic_labels = [
            str(label).strip()
            for label in semantic_labels
            if str(label).strip()
        ]

        suggested_hook = str(
            result.get(
                "suggested_hook",
                "",
            )
        ).strip()

        return {
            "semantic_score": semantic_score,
            "novelty_score": novelty_score,
            "clarity_score": clarity_score,
            "hook_score": hook_score,
            "semantic_labels": semantic_labels,
            "suggested_hook": suggested_hook,
        }

    except json.JSONDecodeError:

        logger.exception(
            "Invalid JSON returned by Groq | text=%r",
            text,
        )

        return _default_scores()

    except Exception:

        logger.exception(
            "LLM segment analysis failed | text=%r",
            text,
        )

        return _default_scores()


def _normalize_score(
    value: Any,
) -> float:

    try:

        score = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def _default_scores() -> dict[str, Any]:

    return {
        "semantic_score": 0.0,
        "novelty_score": 0.0,
        "clarity_score": 0.0,
        "hook_score": 0.0,
        "semantic_labels": [],
        "suggested_hook": "",
    }