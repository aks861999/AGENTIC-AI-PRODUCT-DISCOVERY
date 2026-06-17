from sentence_transformers import SentenceTransformer

# 1. Load the exact model specified in your .env configuration
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 2. Encode the query string with L2 normalisation enabled
query = "paint"
vector_array = model.encode(query, normalize_embeddings=True).tolist()

# 3. View the 384-dimensional float array
print(f"Vector Length: {len(vector_array)}")
print(f"First 3 elements: {vector_array[:3]}")
print(vector_array)