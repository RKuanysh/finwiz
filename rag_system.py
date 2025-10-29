import os
import faiss
import numpy as np
import pickle
from pypdf import PdfReader
from google import genai

# Relative cache path (works locally)
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class RAGSystem:
    def __init__(self, embed_model="text-embedding-004", chat_model="gemini-2.5-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GOOGLE_API_KEY environment variable")
        self.client = genai.Client(api_key=api_key)
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.index = None
        self.chunks = None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _chunk_text(self, text, chunk_size=800, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _embed_content(self, content):
        result = self.client.models.embed_content(
            model=self.embed_model,
            contents=[{"parts": [{"text": content}]}],
        )
        return np.array(result.embeddings[0].values, dtype="float32")

    # -----------------------------
    # Index management
    # -----------------------------
    def build_index(self, pdf_path):
        index_path = os.path.join(CACHE_DIR, "faiss.index")
        chunks_path = os.path.join(CACHE_DIR, "chunks.pkl")

        print(f"📄 Reading PDF from: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.chunks = self._chunk_text(text)

        print(f"🔹 Generating embeddings for {len(self.chunks)} chunks...")
        embeddings = [self._embed_content(chunk) for chunk in self.chunks]
        embeddings_np = np.array(embeddings)
        dim = embeddings_np.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_np)

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ FAISS index and chunks saved in {CACHE_DIR}")

    def load_index(self):
        index_path = os.path.join(CACHE_DIR, "faiss.index")
        chunks_path = os.path.join(CACHE_DIR, "chunks.pkl")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Cache not found. Please run build_index first.")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"✅ Cache loaded from {CACHE_DIR}")

    def prepare(self, pdf_path):
        """Automatically load or build index."""
        index_path = os.path.join(CACHE_DIR, "faiss.index")
        chunks_path = os.path.join(CACHE_DIR, "chunks.pkl")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.load_index()
        else:
            self.build_index(pdf_path)

    # -----------------------------
    # Retrieval and Q&A
    # -----------------------------
    def _retrieve_similar(self, query, k=3):
        q_vec = self._embed_content(query).reshape(1, -1)
        _, indices = self.index.search(q_vec, k)
        return [self.chunks[i] for i in indices[0]]

    def answer_question(self, query):
        if self.index is None or self.chunks is None:
            raise RuntimeError("❌ No index loaded. Run prepare() first.")
        retrieved = self._retrieve_similar(query)
        context = "\n\n".join(retrieved)
        prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
        response = self.client.models.generate_content(
            model=self.chat_model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip()
