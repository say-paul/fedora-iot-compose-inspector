"""Small reusable helpers that do not depend on Fedora services or an AI provider."""


def compact_text(value, limit):
    """Collapse whitespace and cap text for a human-readable report field."""
    compacted = " ".join(str(value).split())
    return compacted[:limit] + "..." if len(compacted) > limit else compacted


def extract_signal_windows(content, signal_pattern, context_lines=1, max_windows=4, max_chars=7000):
    """Return small, merged windows around signal lines from a potentially large log.

    If no signal is present, return the log tail, which commonly contains the
    terminal failure. The caller supplies the signal pattern so this helper remains
    usable for logs from any system.
    """
    lines = content.splitlines()
    signal_indexes = [
        index for index, line in enumerate(lines) if signal_pattern.search(line)
    ][-max_windows:]

    if not signal_indexes:
        return content[-max_chars:]

    ranges = []
    for index in signal_indexes:
        start = max(0, index - context_lines)
        end = min(len(lines), index + context_lines + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    windows = []
    for start, end in ranges:
        windows.extend(lines[start:end])
        windows.append("... [next error window] ...")
    excerpt = "\n".join(windows[:-1])
    return excerpt[-max_chars:]


def prioritize_items(items, preferred_items):
    """Move requested items to the front while retaining each original item once."""
    preferred = [item for item in preferred_items if item in items]
    remaining = [item for item in items if item not in preferred]
    return preferred + remaining
