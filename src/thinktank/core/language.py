"""Session language → the name used in prompt instructions.

``SessionConfig.language`` is stored as a short code (``"hu"``, ``"en"``, …).
Models follow "Speak in Hungarian." more reliably than "Speak in hu.", so the
common codes are mapped to their English language names; anything unknown is
passed through unchanged (so a free-text value like ``"Hungarian"`` also works).
"""

from __future__ import annotations

_NAMES: dict[str, str] = {
    "hu": "Hungarian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "tr": "Turkish",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
}


def language_name(code: str | None) -> str:
    """Return a prompt-friendly language name for ``code`` ('' when unset)."""
    if not code or not code.strip():
        return ""
    value = code.strip()
    base = value.lower().replace("_", "-").split("-", 1)[0]
    return _NAMES.get(base, value)


def language_instruction(code: str | None) -> str:
    """One-line instruction for side calls; '' when no language is set."""
    name = language_name(code)
    return f"Write your answer in {name}." if name else ""
