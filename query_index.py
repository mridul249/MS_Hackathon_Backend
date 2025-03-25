import pickle
import numpy as np

import faiss
import numpy as np
import pickle
import time

from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for the given text using Sentence Transformers."""
    # Encode the text and return as a list of floats
    return model.encode(text).tolist()

def load_index():
    with open("faiss_index.pkl", "rb") as f:
        index, chunks = pickle.load(f)
    return index, chunks

def query_index(query, k=3):
    index, chunks = load_index()
    query_vector = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    user_question = "What are my rights if a product is defective?"
    relevant_chunks = query_index(user_question, k=3)
    print("Top relevant chunks:")
    for chunk in relevant_chunks:
        print(chunk)
        print("-----")
