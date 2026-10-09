"""Parse MOSS's full decoder text without relying on the SDK's lossy regex.

Timestamp markers can be shared by adjacent turns. Estimated or invalid timing
and possible decoder repetition are retained as review flags, not corrected
words. Raw output remains available alongside every normalized segment.
"""
from __future__ import annotations

import math
import re
from typing import Any

PARSER_VERSION = "markers-v1"
_MARKER = re.compile(r"\[\s*(?:(?P<time>-?\d+(?:\.\d+)?|\.\d+)|(?P<speaker>S\d+))\s*\]", re.I)
_SPECIAL = re.compile(r"<\|[^>]*\|>")
_NONVERBAL = re.compile(
    r"\[\s*(?:sniff(?:ing)?|cough(?:ing)?|laugh(?:ing|ter)?|sigh(?:ing)?|"
    r"breathing|silence|music|noise|inaudible|applause)\s*\]", re.I,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _NONVERBAL.sub("", _SPECIAL.sub("", text))).strip()


def _repeated(text: str) -> bool:
    """Conservative review hint; even long genuine repetitions remain verbatim."""
    words = re.findall(r"\w+", text.casefold())
    for width in range(1, min(8, len(words) // 4) + 1):
        for index in range(width, len(words) - width + 1):
            if words[index:index + width] == words[index - width:index]:
                # Examine aligned runs, rather than counting overlapping matches.
                aligned = index - width
                run = 2
                while aligned >= width and words[aligned:aligned + width] == words[aligned - width:aligned]:
                    run += 1
                    aligned -= width
                if (width == 1 and run >= 8) or (run >= 4 and run * width >= 16):
                    return True
    return False


def parse(text: str, duration: float) -> list[dict[str, Any]]:
    """Keep all lexical turns, including a final turn without an end marker.

    MOSS timing is segment-level. A missing boundary falls back to a bounded
    interval with an explicit flag; this does not establish word alignment.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Decoder window must have a positive finite duration")
    markers = list(_MARKER.finditer(text))
    output: list[dict[str, Any]] = []
    start: float | None = None
    speaker = "SPEAKER_UNKNOWN"
    body = ""
    raw_start = 0
    cursor = 0
    flags: set[str] = set()

    def emit(end: float | None, raw_end: int, extra: set[str] | None = None) -> None:
        spoken = _clean(body)
        if not any(character.isalnum() for character in spoken):
            return
        quality = flags | (extra or set())
        left = start if start is not None else 0.0
        right = end if end is not None else duration
        if start is None:
            quality.add("estimated_start")
        if end is None:
            quality.add("estimated_end")
        if speaker == "SPEAKER_UNKNOWN":
            quality.add("unknown_speaker")
        if right <= left:
            left, right = 0.0, duration
            quality.add("invalid_timing")
        if left < 0 or right > duration or left >= duration or right <= 0:
            quality.add("clipped_timing")
            left, right = max(0.0, min(duration, left)), max(0.0, min(duration, right))
            if right <= left:
                left, right = 0.0, duration
                quality.add("invalid_timing")
        if _repeated(spoken):
            quality.add("possible_repetition")
        output.append({"start": left, "end": right, "speaker_id": speaker,
                       "text": spoken, "rawText": text[raw_start:raw_end].strip(),
                       "qualityFlags": sorted(quality), "alignment": "segment"})

    for index, marker in enumerate(markers):
        body += text[cursor:marker.start()]
        if marker.group("time") is not None:
            when = float(marker.group("time"))
            # Very large decimal markers can overflow to infinity.
            if not math.isfinite(when):
                flags.add("invalid_timing")
                cursor = marker.end()
                continue
            emit(when, marker.end())
            body = ""
            start = when
            raw_start = marker.start()
            flags = set()
        else:
            next_speaker = marker.group("speaker").upper()
            if any(character.isalnum() for character in _clean(body)):
                next_time = next((float(item.group("time")) for item in markers[index + 1:]
                                  if item.group("time") is not None), None)
                emit(next_time, marker.start(), {"estimated_end"})
                body = ""
                raw_start = marker.start()
                flags = {"estimated_start"}
            speaker = next_speaker
        cursor = marker.end()
    body += text[cursor:]
    emit(None, len(text))
    return output


def from_decoder(raw_text: str | None, sdk_segments: list[dict], duration: float) -> list[dict[str, Any]]:
    if raw_text is not None:
        return parse(raw_text, duration)
    # Older adapters/test doubles may supply only SDK rows. Preserve their
    # timing but disclose that the full decoder text could not be inspected.
    output = []
    for row in sdk_segments:
        text = str(row.get("text") or "")
        if any(marker.group("time") is not None for marker in _MARKER.finditer(text)):
            normalized = parse(text, duration)
        else:
            normalized = parse(text, duration)
            for item in normalized:
                item["start"] = max(0.0, min(duration, float(row.get("start", 0))))
                item["end"] = max(item["start"], min(duration, float(row.get("end", duration))))
                item["speaker_id"] = str(row.get("speaker_id") or item["speaker_id"])
                item["qualityFlags"] = [flag for flag in item["qualityFlags"]
                                        if flag not in {"estimated_start", "estimated_end"}]
        for item in normalized:
            item["qualityFlags"] = sorted(set(item["qualityFlags"]) | {"decoder_text_unavailable"})
        output.extend(item for item in normalized if item["end"] > item["start"])
    return output


def review_flags(words: list[dict]) -> list[str]:
    return sorted({flag for word in words for flag in word.get("qualityFlags", [])})


def needs_content_review(words: list[dict]) -> bool:
    return "possible_repetition" in review_flags(words)
