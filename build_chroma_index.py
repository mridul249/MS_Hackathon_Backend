import os
import chromadb
import numpy as np
import pickle
from chromadb.config import Settings  # Only needed if you want to use Settings; in new syntax we pass persist_directory directly.
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for the given text using Sentence Transformers."""
    return model.encode(text).tolist()

def load_chunks(chunks_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        content = f.read()
    # Chunks are separated by "===="
    chunks = [chunk.strip() for chunk in content.split("====") if chunk.strip()]
    return chunks

if __name__ == "__main__":
    chunks_file = os.path.join("extracted_texts", "chunks.txt")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks from '{chunks_file}'.")

    # Initialize a Chroma client
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="legal_documents")

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        emb = get_embedding(chunk)
        ids.append(f"chunk_{i}")
        embeddings.append(emb)
        documents.append(chunk)
        metadatas.append({"source": "pdfs"})  # Optional metadata

    # Add data to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print("Chroma collection built and saved.")
