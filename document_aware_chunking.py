import re

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




def document_aware_chunk(
    text:              str,
    max_chunk_size:    int = 1000
) -> list[dict]:
    """
    Split a structured document (markdown, policy doc, wiki page)
    using its own headers as natural chunk boundaries.

    Returns dicts — not just strings — because metadata matters.
    """
    # Match markdown headers: # Title, ## Section, ### Subsection
    header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

    chunks   = []
    headers  = list(header_pattern.finditer(text))

    for i, header in enumerate(headers):
        # Get section title
        level   = len(header.group(1))       # number of # symbols
        title   = header.group(2).strip()

        # Get section content — everything until next header
        start   = header.end()
        end     = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[start:end].strip()

        # Skip empty sections
        if not content:
            continue

        # If section is too large — recursively chunk it
        if len(content) > max_chunk_size:
            sub_chunks = recursive_chunk(content, chunk_size=max_chunk_size)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "id":      f"{title.lower().replace(' ', '_')}_{j}",
                    "text":    f"{title}\n\n{sub}",    # prepend title for context
                    "metadata": {
                        "section": title,
                        "level":   level,
                        "part":    j + 1
                    }
                })
        else:
            chunks.append({
                "id":   title.lower().replace(" ", "_"),
                "text": f"{title}\n\n{content}",
                "metadata": {
                    "section": title,
                    "level":   level
                }
            })

    return chunks