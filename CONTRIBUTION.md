# ShortsYou NLP Contribution Documentation

## Overview

This document records the contribution made by **mayank1312** to the
`shortsYou` repository. The project is a FastAPI-based ML intelligence
service for ShortsYou. It processes creator videos and transcripts to
produce transcription data, content-quality analysis, emotion signals,
Creator DNA profiles, and trained CPEP prediction models.

The implementation is organized as a feature-oriented Python service:

```text
app/
├── auth/                 API-key authentication
├── core/                 Environment configuration
├── db/                   MongoDB connection
├── features/
│   ├── transcribe/       Audio download, transcription, fillers, gaps
│   ├── analyze/          Embeddings and segment quality analysis
│   ├── emotion/          Segment-level emotion and speech signals
│   ├── creator_dna/      Topic modeling and creator fingerprints
│   └── train_cpep/      CPEP model training and versioning
└── shared/               Logging, callbacks, downloads, API-key rotation
```

## Contribution Summary

The work delivered:

- A production-oriented FastAPI ML service with versioned API routes.
- Asynchronous background processing for expensive ML and media jobs.
- MongoDB persistence for transcription, analysis, emotion, Creator DNA,
  and trained-model results.
- Video/audio downloading and FFmpeg-based media processing.
- Deepgram transcription with timestamped words and language detection.
- Filler-word detection and silence-gap detection/recovery.
- Embedding generation and LLM-assisted segment scoring.
- Emotion classification with speech emphasis and speech-rate metrics.
- Creator DNA generation using LDA topic modeling, topic coherence,
  vocabulary extraction, and LLM topic labels.
- Incremental Creator DNA synchronization that blends new results with an
  existing profile.
- XGBoost-based CPEP training with temporal evaluation and SHAP feature
  importance.
- API-key authentication, Groq key rotation, callback delivery, logging,
  Docker deployment, and a keep-alive workflow.

## Public API

All feature routes are mounted under `/api/v1` and protected by the
`Authorization` header when `API_ACCESS_KEY` is configured.

| Endpoint | Purpose | Processing model |
| --- | --- | --- |
| `GET /health` | Service health check | Synchronous |
| `POST /api/v1/transcribe` | Accept a video/audio URL for transcription | Background job |
| `POST /api/v1/analyze` | Analyze transcript segments | Background job |
| `POST /api/v1/emotion` | Classify emotions for audio segments | Background job |
| `POST /api/v1/generate-dna` | Generate a Creator DNA profile | Background job |
| `POST /api/v1/generate-dna/sync` | Incrementally update an existing Creator DNA profile | Background job |
| `POST /api/v1/train-cpep` | Train a creator or pooled CPEP model | Background job |

Long-running endpoints return an accepted response immediately. Results are
stored in MongoDB and, where configured, sent to the Go backend through
internal callbacks with retry handling.

## Feature Details

### 1. Transcription pipeline

The transcription feature accepts `videoId`, `audioUrl`, an optional
`userId`, and an optional language. It:

1. Sends the original media URL to Deepgram.
2. Downloads and extracts audio locally in parallel.
3. Measures actual media duration with FFprobe.
4. Produces timestamped transcript segments and word probabilities.
5. Detects silence gaps and rechecks suspicious gaps using FFmpeg slices.
6. Detects filler words and includes them in the final response.
7. Saves the result to MongoDB.
8. Sends success or failure data to the configured callback.

This pipeline was optimized to avoid unnecessarily uploading downloaded media
back to the transcription provider.

### 2. Transcript analysis

The analysis feature generates sentence embeddings with the configured
MiniLM/Sentence Transformers model. Each transcript segment is scored with
LLM assistance for:

- Semantic quality
- Novelty
- Clarity
- Hook strength
- Semantic labels
- Suggested hook text

The embedding and scores are persisted in MongoDB and returned through the
analysis callback. The MiniLM model is loaded lazily to avoid exhausting
memory during application startup.

### 3. Emotion extraction

Emotion processing downloads the audio and evaluates each requested segment.
For every segment it calculates:

- Emotion label
- Emotion confidence score
- Speech emphasis score based on RMS energy
- Speech rate in words per minute based on saved transcript timestamps

Temporary audio files are removed after processing, including failure paths.

### 4. Creator DNA

Creator DNA is generated from a creator’s saved transcript library. The
pipeline:

- Builds TF-IDF features from transcript text.
- Fits an LDA topic model with a bounded topic count.
- Computes topic coherence.
- Retries once with more topics when coherence is weak.
- Uses Groq to label topic clusters in one batched request.
- Calculates topic diversity, content depth, speech pace, emotional range,
  hook strength, vocabulary, and topic distributions.
- Saves the resulting profile and sends it through the DNA callback.

The synchronization flow fetches the latest transcripts and the existing
profile, then blends the new profile with the old one. This prevents a single
small batch of videos from changing a creator’s fingerprint too sharply.

Single-transcript handling was added so the DNA path can process a creator
with only one available transcript without failing on an invalid topic-model
configuration.

### 5. CPEP model training

The `/train-cpep` route accepts training clips containing:

- Emotion score
- Semantic score
- Hook score
- Duration
- Views during the first 48 hours

The service trains an XGBoost regressor against `log1p(views_48h)`. It uses
chronological data for the train/test split, calculates Pearson correlation,
RMSE, and R², and attempts to generate SHAP feature importances. Models are
serialized and versioned in MongoDB per creator. A pooled model can be
trained with the `global-model` user identifier and requires a larger sample
set.

## Reliability and Operations

### Authentication and configuration

Configuration is environment-based and includes MongoDB, Groq, callback,
Deepgram-related, model, and API access settings. API-key authentication
accepts either a raw key or a `Bearer`-prefixed value.

### External API resilience

Groq requests use a key rotator. Retryable rate-limit, quota, and server
errors rotate to the next configured key before failing.

### Callback delivery

Callback requests:

- Use the internal callback key when configured.
- Have a bounded timeout.
- Retry failed deliveries up to three times.
- Log permanent delivery failures explicitly.

### Deployment

The service includes:

- A Python 3.11 Docker image.
- FFmpeg and FFprobe installation.
- Render-compatible port handling through `PORT`.
- A GitHub Actions keep-alive workflow that checks `/health` every ten
  minutes to reduce free-tier cold starts.

## Commit Timeline

All commits in the repository history were authored by **mayank1312**.

| Date | Commit | Contribution |
| --- | --- | --- |
| 2026-09-02 | `aa77b1f` | Established the FastAPI service, MongoDB persistence, transcription pipeline, analysis pipeline, media downloader, schemas, and logging. |
| 2026-09-02 | `2154670` | Optimized API results, introduced background workflows, API authentication, Groq key rotation, and processing improvements. |
| 2026-09-02 | `9073778` | Added Docker deployment support with FFmpeg. |
| 2026-09-03 | `555c7a8` | Added callback URL support for asynchronous transcription and analysis jobs. |
| 2026-09-05 | `864f0e7` | Added emotion extraction models, persistence, routes, and service logic. |
| 2026-09-05 | `28756c8` | Reversed the earlier emotion route integration change while retaining the feature implementation. |
| 2026-09-05 | `f23d395` | Added the standalone emotion-service implementation and supporting infrastructure before consolidation. |
| 2026-09-06 | `d7cc9b1` | Removed callback URLs from request inputs, centralized callback configuration, and removed the separate emotion-service copy. |
| 2026-09-28 | `04a4c52` | Added Creator DNA routes, schemas, MongoDB storage, topic modeling, and a test video asset. |
| 2026-09-29 | `0823d33` | Lazy-loaded MiniLM to prevent out-of-memory failures during boot. |
| 2026-09-29 | `0ca1a98` | Added asynchronous Creator DNA generation and profile synchronization. |
| 2026-09-29 | `236ab87` | Completed the second part of the asynchronous DNA flow and adjusted its contracts. |
| 2026-09-29 | `cf1926e` | Added support for Creator DNA generation from a single transcript. |
| 2026-11-09 | `6e6b549` | Added the CPEP ML model training route, validation, XGBoost training, SHAP importances, and model version persistence. |

From the initial service commit through the current `main` branch, the
feature work represents approximately **3,486 added lines and 176 removed
lines across 33 changed files**, in addition to the original service
foundation.

## Current Limitations and Follow-up Areas

These are intentionally documented rather than presented as completed:

- Emotion routes exist in the feature package, but the current application
  entry point does not include the emotion router.
- `evergreens_ratio` is currently returned as `0.0` until the planned
  evergreen classifier and dataset are available.
- CPEP training reports results through logs and MongoDB rather than a
  callback endpoint.
- Background tasks depend on the running application process; a durable queue
  would be a future improvement for stronger retry and execution guarantees.
- Production operation requires the relevant environment variables,
  MongoDB, external model/API credentials, and FFmpeg.

## Files Added or Extended

The main contribution areas are:

- `app/features/transcribe/`
- `app/features/analyze/`
- `app/features/emotion/`
- `app/features/creator_dna/`
- `app/features/train_cpep/`
- `app/shared/api_rotator.py`
- `app/shared/callback.py`
- `app/shared/downloader.py`
- `app/auth/auth.py`
- `Dockerfile`
- `.github/workflows/keep_alive.yml`

