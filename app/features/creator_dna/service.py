import json
import logging
import math
import re
from typing import Any

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.configuration import settings
from app.features.creator_dna.db import creator_dna_db
from app.features.creator_dna.schema import (
    CreatorDNAResponse,
    EmotionalRangeStats,
    TopicWeight,
)
from app.shared.api_rotator import groq_rotator
from app.shared.callback import send_callback


logger = logging.getLogger(__name__)

TOP_WORDS_PER_TOPIC = 10

LABELING_MODEL = "openai/gpt-oss-120b"

MIN_COHERENCE_THRESHOLD = 0.40

SMALL_LIBRARY_TOPIC_COUNT = 20

LARGE_LIBRARY_TOPIC_COUNT = 50

LARGE_LIBRARY_VIDEO_THRESHOLD = 50


def _pick_topic_count(n_documents: int) -> int:

    base = (
        SMALL_LIBRARY_TOPIC_COUNT
        if n_documents < LARGE_LIBRARY_VIDEO_THRESHOLD
        else LARGE_LIBRARY_TOPIC_COUNT
    )

    # Never ask for more topics than documents - LDA becomes
    # meaningless (near one topic per document) below this point,
    # but capping avoids a hard sklearn error on tiny creator
    # libraries.
    return max(2, min(base, n_documents // 2))


def _tokenize(text: str) -> list[str]:

    return re.findall(r"[a-zA-Z']+", text.lower())


def _fit_lda(
    texts: list[str],
    n_topics: int,
):

    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
        stop_words="english",
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )

    lda.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    return lda, vectorizer, tfidf_matrix, feature_names


def _topic_top_words(
    lda,
    feature_names,
    n_words: int = TOP_WORDS_PER_TOPIC,
) -> list[list[str]]:

    topics_words = []

    for topic in lda.components_:

        top_indices = topic.argsort()[::-1][:n_words]

        topics_words.append(
            [feature_names[index] for index in top_indices]
        )

    return topics_words


def _compute_coherence(
    topics_words: list[list[str]],
    tokenized_texts: list[list[str]],
) -> float:

    try:

        dictionary = Dictionary(tokenized_texts)

        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )

        return round(coherence_model.get_coherence(), 4)

    except Exception:

        logger.exception(
            "Topic coherence computation failed - continuing "
            "without a coherence score"
        )

        return 0.0


def _label_topics_with_llm(
    topics_words: list[list[str]],
) -> list[str]:
    """
    One batched Groq call labels ALL topics at once, not one call
    per topic - a 50-topic library would otherwise burn 50 separate
    LLM calls for something this cheap to batch.
    """

    topics_payload = [
        {"topic": index, "words": words}
        for index, words in enumerate(topics_words)
    ]

    prompt = (
        "Each item below is a topic cluster extracted via LDA "
        "topic modeling on a content creator's video transcripts, "
        "identified by its top words.\n\n"
        f"{json.dumps(topics_payload)}\n\n"
        "For each topic, respond with a short 2-4 word "
        "human-readable label (e.g. 'react hooks', 'career "
        "advice'). Return ONLY a JSON array of strings, in the "
        "same order as the topics given, with no explanation."
    )

    try:

        response = groq_rotator.create_chat_completion(
            model=LABELING_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        match = re.search(r"\[.*\]", content, re.DOTALL)
        labels = json.loads(match.group(0)) if match else []

        if (
            isinstance(labels, list)
            and len(labels) == len(topics_words)
        ):
            return [str(label) for label in labels]

        logger.warning(
            "LLM topic labeling returned unexpected shape - "
            "falling back to raw top words"
        )

    except Exception:

        logger.exception(
            "LLM topic labeling failed - falling back to raw "
            "top words per topic"
        )

    return [
        ", ".join(words[:3])
        for words in topics_words
    ]


def _topic_diversity_score(
    weights: list[float],
) -> float:
    """
    Shannon entropy of the topic weight distribution, normalized to
    0-1. Higher = content spread evenly across many topics, lower =
    concentrated on a few dominant ones.
    """

    total = sum(weights)

    if total <= 0:
        return 0.0

    probabilities = [
        weight / total
        for weight in weights
        if weight > 0
    ]

    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
    )

    max_entropy = (
        math.log(len(weights))
        if len(weights) > 1
        else 1.0
    )

    return (
        round(entropy / max_entropy, 4)
        if max_entropy > 0
        else 0.0
    )


def _content_depth_score(
    texts: list[str],
) -> float:
    """
    Rough proxy for content depth: average word count per video,
    normalized against a reasonable ceiling (1500 words) rather
    than a hard-coded arbitrary scale.
    """

    if not texts:
        return 0.0

    word_counts = [len(_tokenize(text)) for text in texts]

    average = sum(word_counts) / len(word_counts)

    ceiling = 1500.0

    return round(min(1.0, average / ceiling), 4)


async def generate_creator_dna(
    request,
) -> CreatorDNAResponse:

    try:
        return await _run_creator_dna(request)

    except Exception as error:

        logger.exception(
            "Creator DNA job failed | job_id=%s | user_id=%s",
            request.job_id,
            request.user_id,
        )

        await send_callback(
            callback_url=settings.DNA_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload={
                "job_id": request.job_id,
                "user_id": request.user_id,
                "error": str(error) or "creator DNA generation failed",
            },
        )

        raise


async def _run_creator_dna(
    request,
) -> CreatorDNAResponse:

    texts = [
        text
        for text in request.all_transcript_texts
        if text and text.strip()
    ]

    logger.info(
        "Starting Creator DNA generation | job_id=%s | user_id=%s | "
        "videos=%d",
        request.job_id,
        request.user_id,
        len(texts),
    )

    if not texts:

        logger.warning(
            "No transcript texts received | job_id=%s | user_id=%s",
            request.job_id,
            request.user_id,
        )

        empty_response = CreatorDNAResponse(
            job_id=request.job_id,
            user_id=request.user_id,
        )

        await send_callback(
            callback_url=settings.DNA_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload={
                "job_id": request.job_id,
                "user_id": request.user_id,
                "topic_distribution": [],
                "dna_vector": [],
                "top_vocabulary": [],
                "emotional_range": {
                    "mean": 0.0,
                    "std": 0.0,
                    "dominant": "unknown",
                },
                "avg_speech_rate": 0.0,
                "evergreens_ratio": 0.0,
                "error": "",
            },
        )

        return empty_response

    # ---------------------------------------------------------
    # LDA TOPIC MODELING
    # ---------------------------------------------------------

    n_topics = _pick_topic_count(len(texts))

    lda, vectorizer, tfidf_matrix, feature_names = _fit_lda(
        texts,
        n_topics,
    )

    topics_words = _topic_top_words(lda, feature_names)

    tokenized_texts = [_tokenize(text) for text in texts]

    coherence_score = _compute_coherence(
        topics_words,
        tokenized_texts,
    )

    # One retry with more topics if coherence is weak - bounded to
    # a single retry so a genuinely small/noisy library doesn't
    # loop forever chasing an unreachable coherence score.
    if (
        coherence_score < MIN_COHERENCE_THRESHOLD
        and n_topics < len(texts)
    ):

        logger.info(
            "Coherence %.2f below threshold %.2f - retrying with "
            "more topics | job_id=%s",
            coherence_score,
            MIN_COHERENCE_THRESHOLD,
            request.job_id,
        )

        n_topics = min(n_topics * 2, len(texts))

        lda, vectorizer, tfidf_matrix, feature_names = _fit_lda(
            texts,
            n_topics,
        )

        topics_words = _topic_top_words(lda, feature_names)

        coherence_score = _compute_coherence(
            topics_words,
            tokenized_texts,
        )

    # ---------------------------------------------------------
    # TOPIC LABELING + WEIGHTS
    # ---------------------------------------------------------

    labels = _label_topics_with_llm(topics_words)

    document_topic_matrix = lda.transform(tfidf_matrix)

    topic_weights = document_topic_matrix.sum(axis=0)

    topic_distribution = sorted(
        [
            TopicWeight(
                topic=index,
                label=labels[index],
                weight=round(
                    float(
                        topic_weights[index] / topic_weights.sum()
                    ),
                    4,
                ),
            )
            for index in range(n_topics)
        ],
        key=lambda item: item.weight,
        reverse=True,
    )

    # ---------------------------------------------------------
    # TOP VOCABULARY (overall creator vocabulary, not per-topic)
    # ---------------------------------------------------------

    aggregate_tfidf = tfidf_matrix.sum(axis=0).A1

    top_vocabulary_indices = aggregate_tfidf.argsort()[::-1][:20]

    top_vocabulary = [
        feature_names[index]
        for index in top_vocabulary_indices
    ]

    # ---------------------------------------------------------
    # CROSS-REFERENCE EXISTING EMOTION/ANALYZE DATA (best effort)
    # ---------------------------------------------------------

    emotion_stats = creator_dna_db.get_emotion_stats(
        request.user_id
    )

    avg_hook_score = creator_dna_db.get_avg_hook_score(
        request.user_id
    )

    # ---------------------------------------------------------
    # 6-DIM DNA VECTOR
    # ---------------------------------------------------------

    dna_vector = [
        _topic_diversity_score(
            [item.weight for item in topic_distribution]
        ),
        round(emotion_stats["std"], 4),
        round(
            min(1.0, emotion_stats["avg_speech_rate"] / 200.0),
            4,
        ),
        _content_depth_score(texts),
        round(avg_hook_score, 4),
        0.0,  # evergreenRatio - classifier not built yet
    ]

    response = CreatorDNAResponse(
        job_id=request.job_id,
        user_id=request.user_id,
        topic_distribution=topic_distribution,
        dna_vector=dna_vector,
        top_vocabulary=top_vocabulary,
        emotional_range=EmotionalRangeStats(
            mean=emotion_stats["mean"],
            std=emotion_stats["std"],
            dominant=emotion_stats["dominant"],
        ),
        avg_speech_rate=emotion_stats["avg_speech_rate"],
        evergreens_ratio=0.0,
        topic_coherence_score=coherence_score,
    )

    logger.info(
        "Creator DNA generation completed | job_id=%s | user_id=%s | "
        "topics=%d | coherence=%.2f",
        request.job_id,
        request.user_id,
        n_topics,
        coherence_score,
    )

    creator_dna_db.save_dna(
        job_id=request.job_id,
        user_id=request.user_id,
        profile=response.model_dump(),
    )

    await send_callback(
        callback_url=settings.DNA_CALLBACK_URL,
        internal_key=settings.INTERNAL_CALLBACK_KEY,
        payload={
            "job_id": request.job_id,
            "user_id": request.user_id,
            "topic_distribution": [
                item.model_dump() for item in topic_distribution
            ],
            "dna_vector": dna_vector,
            "top_vocabulary": top_vocabulary,
            "emotional_range": response.emotional_range.model_dump(),
            "avg_speech_rate": response.avg_speech_rate,
            "evergreens_ratio": response.evergreens_ratio,
            "error": "",
        },
    )

    return response