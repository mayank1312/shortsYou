import asyncio
import logging
import os
import uuid

import httpx


logger = logging.getLogger(__name__)

DOWNLOAD_DIRECTORY = "temp"


async def get_media_duration(
    file_path: str,
) -> float | None:

    try:

        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:

            logger.warning(
                "FFprobe failed | path=%s | stderr=%s",
                file_path,
                stderr.decode(errors="ignore"),
            )

            return None

        duration_text = stdout.decode().strip()

        if not duration_text:
            return None

        return float(duration_text)

    except Exception:

        logger.exception(
            "Failed to determine media duration | path=%s",
            file_path,
        )

        return None


async def slice_audio(
    audio_path: str,
    start: float,
    end: float,
) -> str:
    """
    Cut out [start, end] from an existing audio file using ffmpeg
    and return the path to the new slice.

    Used to re-send a specific window of audio to Deepgram in
    isolation, when the first pass over the full file returned no
    words for that window (a suspected VAD / language-detection
    miss rather than genuine silence).
    """

    os.makedirs(
        DOWNLOAD_DIRECTORY,
        exist_ok=True,
    )

    slice_path = os.path.join(
        DOWNLOAD_DIRECTORY,
        f"{uuid.uuid4()}_slice.wav",
    )

    # Small padding on both sides so we don't clip the first/last
    # syllable right at the boundary.
    padded_start = max(0.0, start - 0.5)
    padded_duration = (end - padded_start) + 0.5

    logger.info(
        "Slicing audio for gap re-check | "
        "source=%s | start=%.3f | end=%.3f | padded_start=%.3f | "
        "padded_duration=%.3f",
        audio_path,
        start,
        end,
        padded_start,
        padded_duration,
    )

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ss",
        str(padded_start),
        "-t",
        str(padded_duration),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        slice_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:

        logger.error(
            "FFmpeg slicing failed | stderr=%s",
            stderr.decode(errors="ignore"),
        )

        raise RuntimeError(
            "Audio slicing failed"
        )

    logger.info(
        "Audio slice created | path=%s | size=%d bytes",
        slice_path,
        os.path.getsize(slice_path),
    )

    return slice_path


async def download_audio(video_url: str) -> str:

    os.makedirs(
        DOWNLOAD_DIRECTORY,
        exist_ok=True,
    )

    video_file = os.path.join(
        DOWNLOAD_DIRECTORY,
        f"{uuid.uuid4()}_video.mp4",
    )

    audio_file = os.path.join(
        DOWNLOAD_DIRECTORY,
        f"{uuid.uuid4()}_audio.wav",
    )

    try:

        # ---------------------------------------------------------
        # DOWNLOAD VIDEO
        # ---------------------------------------------------------

        logger.info(
            "Downloading video | url=%s",
            video_url,
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0)
        ) as client:

            response = await client.get(video_url)
            response.raise_for_status()

            with open(video_file, "wb") as file:
                file.write(response.content)

        video_size = os.path.getsize(video_file)

        logger.info(
            "Video downloaded | path=%s | size=%d bytes",
            video_file,
            video_size,
        )

        # ---------------------------------------------------------
        # VERIFY DOWNLOADED VIDEO DURATION
        # ---------------------------------------------------------

        video_duration = await get_media_duration(
            video_file,
        )

        logger.info(
            "Downloaded video duration | "
            "duration=%s seconds | path=%s",
            (
                round(video_duration, 3)
                if video_duration is not None
                else "unknown"
            ),
            video_file,
        )

        # ---------------------------------------------------------
        # EXTRACT AUDIO
        # ---------------------------------------------------------

        logger.info(
            "Extracting audio using FFmpeg | video=%s",
            video_file,
        )

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            video_file,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:

            logger.error(
                "FFmpeg failed | stderr=%s",
                stderr.decode(errors="ignore"),
            )

            raise RuntimeError(
                "Audio extraction failed"
            )

        audio_size = os.path.getsize(audio_file)

        logger.info(
            "Audio extraction completed | "
            "path=%s | size=%d bytes",
            audio_file,
            audio_size,
        )

        # ---------------------------------------------------------
        # VERIFY EXTRACTED AUDIO DURATION
        # ---------------------------------------------------------

        audio_duration = await get_media_duration(
            audio_file,
        )

        logger.info(
            "Extracted audio duration | "
            "duration=%s seconds | path=%s",
            (
                round(audio_duration, 3)
                if audio_duration is not None
                else "unknown"
            ),
            audio_file,
        )

        # ---------------------------------------------------------
        # COMPARE VIDEO AND AUDIO DURATIONS
        # ---------------------------------------------------------

        if (
            video_duration is not None
            and audio_duration is not None
        ):

            difference = video_duration - audio_duration

            logger.info(
                "Media duration comparison | "
                "video=%.3f seconds | audio=%.3f seconds | "
                "difference=%.3f seconds",
                video_duration,
                audio_duration,
                difference,
            )

            if abs(difference) > 0.5:

                logger.warning(
                    "VIDEO/AUDIO DURATION MISMATCH DETECTED | "
                    "video=%.3f seconds | audio=%.3f seconds | "
                    "difference=%.3f seconds",
                    video_duration,
                    audio_duration,
                    difference,
                )

        return audio_file

    except Exception:

        logger.exception(
            "Video download/audio extraction failed"
        )

        if os.path.exists(audio_file):
            os.remove(audio_file)

        raise

    finally:

        if os.path.exists(video_file):

            os.remove(video_file)

            logger.info(
                "Temporary video removed | path=%s",
                video_file,
            )