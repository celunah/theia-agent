"""Small text-formatting helpers shared by Discord response delivery."""

import re


def _split_pages(text: str, limit: int = 1900) -> list[str]:
    text = text or ""
    if not text:
        return [""]
    pages: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        pages.append(remaining[:split_at])
        remaining = remaining[split_at:]
    pages.append(remaining)
    return pages


def _split_markdown_pages(text: str, limit: int = 1024) -> list[str]:
    """Split bounded pages at Markdown headings when a heading fits."""
    text = text or ""
    if not text:
        return [""]

    sections: list[str] = []
    section_start = 0
    offset = 0
    active_fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        fence = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if active_fence is not None:
            if fence is not None:
                marker = fence.group(1)
                if (
                    marker[0] == active_fence[0]
                    and len(marker) >= active_fence[1]
                    and not line[fence.end() :].strip()
                ):
                    active_fence = None
        elif fence is not None:
            marker = fence.group(1)
            active_fence = (marker[0], len(marker))
        elif offset > section_start and re.match(r" {0,3}#{1,6}[ \t]+\S", line):
            sections.append(text[section_start:offset])
            section_start = offset
        offset += len(line)
    sections.append(text[section_start:])

    pages: list[str] = []
    current = ""
    for section in sections:
        if len(section) > limit:
            if current:
                pages.append(current)
                current = ""
            section_pages = _split_pages(section, limit)
            pages.extend(section_pages[:-1])
            current = section_pages[-1]
        elif len(current) + len(section) <= limit:
            current += section
        else:
            pages.append(current)
            current = section
    if current or not pages:
        pages.append(current)
    return pages


def _format_thought_duration(seconds: float) -> str:
    if seconds < 1:
        return "Thought for less than a second"
    elapsed = max(1, int(seconds))
    if elapsed < 60:
        unit = "second" if elapsed == 1 else "seconds"
        return f"Thought for {elapsed} {unit}"
    minutes, remainder = divmod(elapsed, 60)
    minute_unit = "minute" if minutes == 1 else "minutes"
    if remainder == 0:
        return f"Thought for {minutes} {minute_unit}"
    second_unit = "second" if remainder == 1 else "seconds"
    return f"Thought for {minutes} {minute_unit} and {remainder} {second_unit}"
