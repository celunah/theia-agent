"""Small text-formatting helpers shared by Discord response delivery."""


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
        if split_at <= 0:
            split_at = limit
        pages.append(remaining[:split_at])
        remaining = remaining[split_at:]
    pages.append(remaining)
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
