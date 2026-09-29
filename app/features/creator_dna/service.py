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

LABELING_MODEL = "openai/gpt-oss-120b"

TOP_WORDS_PER_TOPIC = 10

MIN_COHERENCE_THRESHOLD = 0.40

SMALL_LIBRARY_TOPIC_COUNT = 20

LARGE_LIBRARY_TOPIC_COUNT = 50

LARGE_LIBRARY_VIDEO_THRESHOLD = 50

# How much weight the EXISTING profile keeps vs the newly-computed
# batch when syncing. A creator's DNA should be a slowly-shifting
# fingerprint, not something that swings wildly every time a new
# batch of videos is processed. Tune these if that assumption is
# wrong for how this actually gets used.
SYNC_OLD_PROFILE_WEIGHT = 0.7
SYNC_NEW_PROFILE_WEIGHT = 0.3

TOP_VOCABULARY_CAP = 20


def _pick_topic_count(n_documents: int) -> int:

    base = (
        SMALL_LIBRARY_TOPIC_COUNT
        if n_documents < LARGE_LIBRARY_VIDEO_THRESHOLD
        else LARGE_LIBRARY_TOPIC_COUNT
    )

    # Never ask for more topics than documents - LDA becomes
    # meaningless (near one topic per document) below this point,
    # but capping avoids a hard sklearn error on tiny creator
    # libraries. Also cap at half the document count so a small
    # batch doesn't fragment into near-duplicate topics.
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


def _blend_dna_vector(
    old_vector: list[float] | None,
    new_vector: list[float],
) -> list[float]:

    if not old_vector or len(old_vector) != len(new_vector):
        return new_vector

    return [
        round(
            SYNC_OLD_PROFILE_WEIGHT * old_value
            + SYNC_NEW_PROFILE_WEIGHT * new_value,
            4,
        )
        for old_value, new_value in zip(old_vector, new_vector)
    ]


def _blend_topic_distribution(
    old_topics: list[dict[str, Any]] | None,
    new_topics: list[TopicWeight],
) -> list[TopicWeight]:
    """
    Merges by normalized label text (lowercased, stripped) - LDA
    topic indices aren't stable across separate fits, so label
    matching is the only reasonably stable join key available.
    Topics with genuinely new wording are treated as new topics
    rather than merged, which is a real limitation of this
    approach worth knowing about, not a bug.
    """

    if not old_topics:
        return new_topics

    merged: dict[str, dict[str, Any]] = {}

    for old_topic in old_topics:

        key = old_topic.get("label", "").strip().lower()

        if not key:
            continue

        merged[key] = {
            "topic": old_topic.get("topic", 0),
            "label": old_topic.get("label", ""),
            "weight": (
                SYNC_OLD_PROFILE_WEIGHT
                * old_topic.get("weight", 0.0)
            ),
        }

    for new_topic in new_topics:

        key = new_topic.label.strip().lower()

        contribution = SYNC_NEW_PROFILE_WEIGHT * new_topic.weight

        if key in merged:
            merged[key]["weight"] += contribution
            # Prefer the freshest label wording/index on a match.
            merged[key]["label"] = new_topic.label
            merged[key]["topic"] = new_topic.topic
        else:
            merged[key] = {
                "topic": new_topic.topic,
                "label": new_topic.label,
                "weight": contribution,
            }

    total_weight = sum(
        item["weight"] for item in merged.values()
    )

    if total_weight <= 0:
        return new_topics

    return sorted(
        [
            TopicWeight(
                topic=item["topic"],
                label=item["label"],
                weight=round(item["weight"] / total_weight, 4),
            )
            for item in merged.values()
        ],
        key=lambda item: item.weight,
        reverse=True,
    )


def _blend_top_vocabulary(
    old_vocabulary: list[str] | None,
    new_vocabulary: list[str],
) -> list[str]:

    combined: list[str] = []

    for word in new_vocabulary + (old_vocabulary or []):

        if word not in combined:
            combined.append(word)

        if len(combined) >= TOP_VOCABULARY_CAP:
            break

    return combined


async def generate_creator_dna(
    request,
) -> CreatorDNAResponse:
    """
    The original /generate-dna path - Abbas sends the transcript
    texts directly, no fetching, no blending with a prior profile.
    """

    try:

        return await _run_creator_dna(
            job_id=request.job_id,
            user_id=request.user_id,
            texts=request.all_transcript_texts,
            old_profile=None,
        )

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


async def generate_creator_dna_sync(
    user_id: str,
) -> None:
    """
    The new /generate-dna/sync path. Abbas sends only userId - we:
      1. fetch this creator's latest 5 transcripts from Abbas's own
         database (truncated to 250 words each),
      2. fetch whatever DNA profile already exists for this user,
      3. run the normal LDA pipeline on the new batch,
      4. blend the new result into the existing profile rather than
         overwriting it outright,
      5. save + callback.

    This is fired from a background task - there is no HTTP
    response left to raise into, so every failure path here ends in
    a callback, not an exception bubbling up silently.
    """

    logger.info(
        "Starting Creator DNA sync | user_id=%s",
        user_id,
    )

    try:

        texts = creator_dna_db.get_latest_transcript_texts(
            user_id=user_id,
        )

        if not texts:

            logger.warning(
                "No transcripts found for user_id=%s - nothing to "
                "sync yet",
                user_id,
            )

            await send_callback(
                callback_url=settings.DNA_CALLBACK_URL,
                internal_key=settings.INTERNAL_CALLBACK_KEY,
                payload={
                    "user_id": user_id,
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

            return

        old_profile = creator_dna_db.get_latest_dna_profile(
            user_id
        )

        await _run_creator_dna(
            job_id=None,
            user_id=user_id,
            texts=texts,
            old_profile=old_profile,
        )

    except Exception as error:

        logger.exception(
            "Creator DNA sync failed | user_id=%s",
            user_id,
        )

        await send_callback(
            callback_url=settings.DNA_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload={
                "user_id": user_id,
                "error": str(error) or "creator DNA sync failed",
            },
        )


async def _run_creator_dna(
    job_id: str | None,
    user_id: str,
    texts: list[str],
    old_profile: dict[str, Any] | None,
) -> CreatorDNAResponse:

    texts = [
        text
        for text in texts
        if text and text.strip()
    ]

    logger.info(
        "Running Creator DNA generation | job_id=%s | user_id=%s | "
        "videos=%d | syncing_with_existing=%s",
        job_id,
        user_id,
        len(texts),
        bool(old_profile),
    )

    if not texts:

        logger.warning(
            "No transcript texts received | job_id=%s | user_id=%s",
            job_id,
            user_id,
        )

        empty_response = CreatorDNAResponse(
            job_id=job_id,
            user_id=user_id,
        )

        await send_callback(
            callback_url=settings.DNA_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload={
                "job_id": job_id,
                "user_id": user_id,
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
            job_id,
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

    new_topic_distribution = sorted(
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

    new_top_vocabulary = [
        feature_names[index]
        for index in top_vocabulary_indices
    ]

    # ---------------------------------------------------------
    # CROSS-REFERENCE EXISTING EMOTION/ANALYZE DATA (best effort)
    # ---------------------------------------------------------

    emotion_stats = creator_dna_db.get_emotion_stats(user_id)

    avg_hook_score = creator_dna_db.get_avg_hook_score(user_id)

    # ---------------------------------------------------------
    # 6-DIM DNA VECTOR
    # ---------------------------------------------------------

    new_dna_vector = [
        _topic_diversity_score(
            [item.weight for item in new_topic_distribution]
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

    # ---------------------------------------------------------
    # BLEND WITH EXISTING PROFILE (sync path only - old_profile is
    # always None on the original /generate-dna path)
    # ---------------------------------------------------------

    if old_profile:

        dna_vector = _blend_dna_vector(
            old_profile.get("dna_vector"),
            new_dna_vector,
        )

        topic_distribution = _blend_topic_distribution(
            old_profile.get("topic_distribution"),
            new_topic_distribution,
        )

        top_vocabulary = _blend_top_vocabulary(
            old_profile.get("top_vocabulary"),
            new_top_vocabulary,
        )

    else:

        dna_vector = new_dna_vector
        topic_distribution = new_topic_distribution
        top_vocabulary = new_top_vocabulary

    response = CreatorDNAResponse(
        job_id=job_id,
        user_id=user_id,
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
        "topics=%d | coherence=%.2f | blended_with_existing=%s",
        job_id,
        user_id,
        n_topics,
        coherence_score,
        bool(old_profile),
    )

    creator_dna_db.save_dna(
        job_id=job_id,
        user_id=user_id,
        profile=response.model_dump(),
    )

    await send_callback(
        callback_url=settings.DNA_CALLBACK_URL,
        internal_key=settings.INTERNAL_CALLBACK_KEY,
        payload={
            "job_id": job_id,
            "user_id": user_id,
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
