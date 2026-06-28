def recursive_chunk(
    text:       str,
    chunk_size: int = 500,
    overlap:    int = 50
) -> list[str]:
    """
    Split text respecting natural boundaries.

    chunk_size: target max characters per chunk
    overlap:    how many characters to repeat between chunks
                — prevents answers being split at boundaries
    """
    # Separators tried in order — from coarsest to finest
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    def split_with_separator(text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        return text.split(separator)

    def chunk_recursive(text: str, separators: list[str]) -> list[str]:
        chunks     = []
        separator  = separators[0]
        remaining  = separators[1:]
        splits     = split_with_separator(text, separator)
        current    = ""

        for split in splits:
            # Re-add separator (we split on it so it's gone)
            piece = split + separator if separator != "" else split

            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current.strip())
                # If single piece is too large — go deeper
                if len(piece) > chunk_size and remaining:
                    chunks.extend(chunk_recursive(piece, remaining))
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = chunk_recursive(text, separators)

    # Add overlap — repeat end of previous chunk at start of next
    # This is what prevents the "answer split at boundary" problem
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail   = raw_chunks[i - 1][-overlap:]  # last N chars of previous
        overlapped.append(prev_tail + raw_chunks[i]) # prepend to current
    return overlapped



policy_text = """
Leave Policy

Full-time employees receive 24 days of paid annual leave per year.
Leave must be requested at least 3 days in advance.
Emergency medical leave is an exception to this rule.

Unused leave can be carried forward, but only up to 12 days maximum.
Leave beyond 12 days is forfeited at year end.

Probationary employees — those in their first 90 days — receive
only 12 days of pro-rated leave for their first year.
"""

chunks = recursive_chunk(policy_text, chunk_size=200, overlap=30)

for i, chunk in enumerate(chunks, 1):
    print(f"\n── Chunk {i} ({len(chunk)} chars) ──")
    print(chunk)