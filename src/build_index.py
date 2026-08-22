import os
import numpy as np
import faiss

# 1. Load embeddings

embeddings_file = "../embeddings/embeddings_1k.npy"
embeddings = np.load(embeddings_file).astype("float32")
print("Embeddings shape:", embeddings.shape)

# 2. Get vector dimension
dimension = embeddings.shape[1]
print("Vector dimension:", dimension)

# 3. Create FAISS index (IP = cosine similarity => inner proiduct of normalised vectors)
index = faiss.IndexFlatIP(dimension)

# 4. Add embeddings to the index
index.add(embeddings)
print("Index contains:", index.ntotal, "vectors")

# 5. Save the FAISS index

os.makedirs("../faiss_index", exist_ok=True)
index_file = "../faiss_index/index_1k.faiss"

faiss.write_index(index, index_file)
print("Index saved to:", index_file)