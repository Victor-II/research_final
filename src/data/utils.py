def find_span_indices(tokens: list[str], text: str) -> list[int] | None:
    span_tokens = text.split()
    n = len(span_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == span_tokens:
            return list(range(i, i + n))
    lower_tokens = [t.lower() for t in tokens]
    lower_span = [t.lower() for t in span_tokens]
    for i in range(len(lower_tokens) - n + 1):
        if lower_tokens[i:i + n] == lower_span:
            return list(range(i, i + n))
    return None
