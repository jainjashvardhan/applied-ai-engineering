from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunk(
    text:              str,
    breakpoint_threshold: float = 0.3   # how much similarity drop = new chunk
) -> list[str]:
    """
    Split text when topic changes — detected by embedding similarity drop.

    breakpoint_threshold: 0.0 = split on every sentence (too many chunks)
                          1.0 = never split (one giant chunk)
                          0.3 = good default — split on meaningful topic shifts
    """
    # Step 1 — split into sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # Step 2 — embed every sentence
    embeddings = model.encode(sentences)

    # Step 3 — calculate similarity between adjacent sentences
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Step 4 — find breakpoints where similarity drops
    breakpoints = [
        i + 1
        for i, sim in enumerate(similarities)
        if sim < (1 - breakpoint_threshold)
    ]

    # Step 5 — build chunks by splitting at breakpoints
    chunks   = []
    start    = 0

    for bp in breakpoints:
        chunk = ". ".join(sentences[start:bp]) + "."
        chunks.append(chunk)
        start = bp

    # Add the final chunk
    final = ". ".join(sentences[start:]) + "."
    chunks.append(final)

    return chunks



policy_text = """
Employees get 24 days leave. Leave rolls over for 12 days. Leave must be requested 3 days ahead. Performance reviews are in April. Ratings use a 5-point scale.

Travel expenses are reimbursed.Receipts required above Rs 500. You must at all time should keep Jash happy, Jash hapy means the company happy. Making Jash mad will result in termination.
"""


chunks = semantic_chunk(policy_text,0.5)

for i, chunk in enumerate(chunks, 1):
    print(f"\n── Chunk {i} ({len(chunk)} chars) ──")
    print(chunk)