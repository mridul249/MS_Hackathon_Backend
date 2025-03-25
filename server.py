# server.py
import os
import requests
import chromadb
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.getenv("PORT", 5000))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Sentence Transformer model (must be identical to the one used during indexing)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for the given text using Sentence Transformers."""
    return model.encode(text).tolist()

# Initialize Chroma client.
client = chromadb.Client()
collection = client.get_or_create_collection(name="legal_documents")

def query_chroma(query, n_results=3):
    """Retrieve the top n_results relevant chunks from the Chroma collection."""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )
    return results["documents"][0]

# Azure GPT‑4 API details (update with your actual Azure API key and endpoint)
AZURE_API_KEY = "Fj1KPt7grC6bAkNja7daZUstpP8wZTXsV6Zjr2FOxkO7wsBQ5SzQJQQJ99BCACHYHv6XJ3w3AAAAACOGL3Xg"  # Replace with your Azure API key
GPT4_ENDPOINT = "https://ai-aihackthonhub282549186415.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"

def call_gpt4_with_messages(messages):
    """Call the Azure GPT‑4 endpoint with the provided messages."""
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    payload = {
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.5
    }
    response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code} {response.text}"
    data = response.json()
    return data["choices"][0]["message"]["content"]

@app.route("/chat", methods=["POST"])
def chat():
    req_data = request.get_json()
    user_question = req_data.get("question")
    history = req_data.get("history", [])  # history is an array of {role, content}
    
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    # Retrieve relevant legal context from ChromaDB using the current user question
    relevant_chunks = query_chroma(user_question, n_results=3)
    legal_context = "\n\n".join(relevant_chunks)

    # Construct the conversation messages for GPT‑4
    messages = [
        {
            "role": "system",
            "content": "You are a helpful legal assistant specialized in consumer rights. Use the provided legal context to answer the user's question."
        },
        {
            "role": "system",
            "content": f"Legal Context:\n{legal_context}"
        }
    ]
    # Append past conversation history (if any)
    if history:
        messages.extend(history)
    # Append the new user message
    messages.append({"role": "user", "content": user_question})
    
    # Call Azure GPT‑4
    answer = call_gpt4_with_messages(messages)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
