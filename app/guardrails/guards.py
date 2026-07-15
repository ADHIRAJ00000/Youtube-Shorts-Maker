"""Guardrails: enforce output limits in code (defense in depth).

Prompts ask the model to respect limits; these functions *guarantee* them. Each
`enforce_*` returns a fixed value that satisfies the limit; each `*_violations`
returns human-readable violations so a node can trigger a single corrective
retry before falling back to a hard trim.
"""

from __future__ import annotations

MAX_TITLE_CHARS = 60
MAX_TAGS = 15
MAX_HASHTAGS = 8
MAX_HOOK_WORDS = 12
MAX_THUMB_WORDS = 4

# Char budget for any single agent prompt (~4 chars/token). The content map is
# already token-bounded; this caps the free-form overview we build from it.
MAX_PROMPT_CHARS = 8000


# --------------------------------------------------------------------------- #
# Input capping
# --------------------------------------------------------------------------- #
def cap_text(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …[truncated]"


# --------------------------------------------------------------------------- #
# Titles
# --------------------------------------------------------------------------- #
def title_violations(titles: list[str]) -> list[str]:
    return [t for t in titles if len(t) > MAX_TITLE_CHARS]


def trim_title(title: str) -> str:
    title = title.strip()
    if len(title) <= MAX_TITLE_CHARS:
        return title
    # Cut on a word boundary within the limit.
    cut = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
    return (cut or title[:MAX_TITLE_CHARS]).strip()


def enforce_titles(titles: list[str]) -> list[str]:
    return [trim_title(t) for t in titles if t.strip()]


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #
def hook_word_count(text: str) -> int:
    return len(text.split())


def trim_hook(text: str) -> str:
    words = text.strip().split()
    if len(words) <= MAX_HOOK_WORDS:
        return text.strip()
    return " ".join(words[:MAX_HOOK_WORDS])


def hook_violations(texts: list[str]) -> list[str]:
    return [t for t in texts if hook_word_count(t) > MAX_HOOK_WORDS]


# --------------------------------------------------------------------------- #
# Thumbnail text
# --------------------------------------------------------------------------- #
def enforce_thumbnail_text(phrases: list[str]) -> list[str]:
    out: list[str] = []
    for p in phrases:
        words = p.strip().split()
        if not words:
            continue
        out.append(" ".join(words[:MAX_THUMB_WORDS]).upper())
    return out


# --------------------------------------------------------------------------- #
# Tags / hashtags
# --------------------------------------------------------------------------- #
def enforce_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t = t.strip().lstrip("#").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) >= MAX_TAGS:
            break
    return out


def enforce_hashtags(hashtags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for h in hashtags:
        h = h.strip()
        if not h:
            continue
        if not h.startswith("#"):
            h = "#" + h.lstrip("#")
        # Collapse internal whitespace.
        h = "#" + "".join(h[1:].split())
        key = h.lower()
        if h != "#" and key not in seen:
            seen.add(key)
            out.append(h)
        if len(out) >= MAX_HASHTAGS:
            break
    return out
