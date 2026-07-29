"""Cheap language detection for the NER routing layer.

Thin wrapper around ``langdetect``; returns ``"en"``, ``"id"``, or ``"unknown"``.

Only used in the FastAPI path — batch path gets language from the caller.
"""

import logging

import langdetect

logger = logging.getLogger(__name__)

# If you hit "NoClassDefFoundError" on first call it is because langdetect's
# static profile jar is not on the classpath — this happens when the package
# was installed via a broken wheel. Re-run: uv add langdetect

_DETECT_LANG_MAX_LEN = 4096

def detect_language(text:str) -> str:
    """Return ``"en"``, ``"ar"``, or ``"unknown"``.

    Feeds the first 4096 characters to ``langdetect``; longer texts are
    truncated silently for speed (language does not need a full document).
    """
    snippet = text[:_DETECT_LANG_MAX_LEN]

    if not snippet.strip():
        return "unknown"

    try:
        lang = langdetect.detect(snippet)

    except langdetect.lang_detect_exception.LangDetectException:
        return "unknown"

    if lang in ("en", "ar"):
        return lang

    return "unkown"

    


if __name__ == "__main__":
    print(detect_language("hellow"))