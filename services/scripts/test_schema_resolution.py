# services/scripts/test_schema_resolution.py
# Run: set PYTHONPATH=services && python -m scripts.test_schema_resolution

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

CACHE_PATH = Path(__file__).resolve().parents[1] / "api" / "schema_cache.npz"

def cosine_sim(a, b):
    return float(np.dot(a, b))  # already L2-normalised

def test_resolution(extracted_name: str, extracted_value: str = "Moth Gray"):
    data = np.load(CACHE_PATH, allow_pickle=True)
    names = list(data["names"])
    embeddings = data["embeddings"].astype(np.float32)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    query_vec = model.encode(extracted_name, normalize_embeddings=True)
    scores = embeddings @ query_vec  # shape: (N,)

    # Top-10 candidates
    top10_idx = scores.argsort()[-10:][::-1]

    print(f"\n{'='*60}")
    print(f"Resolving: '{extracted_name}'  (value='{extracted_value}')")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Score':<8} {'Attribute Name'}")
    print(f"{'-'*60}")
    for rank, i in enumerate(top10_idx, 1):
        marker = " ← TARGET" if names[i] == "Color/Finish" else ""
        threshold_marker = " ✗ BELOW 0.65" if scores[i] < 0.65 else " ✓"
        print(f"{rank:<5} {scores[i]:.4f}{threshold_marker:<16} {names[i]}{marker}")

    # Check if "Color/Finish" is in the index at all
    print(f"\n{'='*60}")
    if "Color/Finish" in names:
        idx = names.index("Color/Finish")
        print(f"'Color/Finish' EXISTS in cache at index {idx}")
        print(f"  Score vs '{extracted_name}': {scores[idx]:.4f}")
        print(f"  Threshold (EMBED_THRESHOLD): 0.65")
        if scores[idx] >= 0.65:
            print(f"  RESULT: ✓ Would be ACCEPTED as candidate")
        else:
            print(f"  RESULT: ✗ REJECTED — score {scores[idx]:.4f} < 0.65 threshold")
            print(f"  FIX: Lower EMBED_THRESHOLD to {scores[idx] - 0.01:.2f} in SchemaLinker")
    else:
        print("'Color/Finish' NOT FOUND in cache — check your index attribute names!")
        # Show similar names
        print("\nNames containing 'color' (case-insensitive):")
        for n in names:
            if "color" in n.lower():
                print(f"  '{n}'")

if __name__ == "__main__":
    test_resolution("Color", "Moth Gray")
    test_resolution("color", "moth gray")   # lowercase variant