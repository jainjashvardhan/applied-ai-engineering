# understand_embeddings.py
# Run this once just to SEE what embeddings are

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a free embedding model
# Downloads ~90MB on first run — normal, only happens once
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── WHAT DO EMBEDDINGS LOOK LIKE? ─────────────────────

text = "The candidate has strong Python skills"
embedding = model.encode(text)

print(f"Text   : {text}")
print(f"Shape  : {embedding.shape}")        # (384,) — 384 numbers
print(f"First 5: {embedding[:50]}")          # [-0.03,  0.08, -0.02, ...]
print(f"Type   : {type(embedding)}\n")      # numpy array

# ── SIMILAR SENTENCES = SIMILAR NUMBERS ───────────────

sentences = [
    "The candidate has strong Python skills",    # original
    "The applicant is proficient in Python",     # similar meaning
    "She codes well in Python",                  # similar meaning
    "The weather in Mumbai is humid",            # completely different
    "Looking for a senior backend engineer",     # related to jobs, not Python
]

embeddings = model.encode(sentences)

# Cosine similarity — measures how "close" two vectors are
# 1.0 = identical meaning, 0.0 = completely unrelated
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

reference = embeddings[0]  # "The candidate has strong Python skills"

print("Similarity to: 'The candidate has strong Python skills'")
print("─" * 55)
for i, sentence in enumerate(sentences):
    score = cosine_similarity(reference, embeddings[i])
    bar   = "█" * int(score * 20)
    print(f"{score:.3f}  {bar:20s}  {sentence[:45]}")