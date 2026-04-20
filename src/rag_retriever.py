import os
import json
import pickle
import numpy as np
import faiss
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from src.midi_dataset import MIDIDataset

@dataclass
class MIDISnippet:
    file_path: str
    description: str   # e.g. "melancholic Chopin nocturne, slow, D minor"
    mood_tags: list[str]
    metadata: dict

class RAGRetriever:
    """
    Builds and queries a FAISS vector index of MIDI phrase embeddings.
    Used to retrieve stylistically relevant MIDI snippets given a mood/style query.
    """

    def __init__(self, index_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        print("Loading embedding model...")
        self.model = SentenceTransformer(embedding_model)
        print("Model loaded.")
        self.index = None
        self.snippets = []

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def _make_description(self, midi_path: str, metadata: dict) -> str:
        """
        Auto-generates a text description from metadata for embedding.
        e.g. "piano piece in G major, tempo 120 BPM, moderate velocity"
        """
        key = metadata.get("key", "unknown key")
        tempo = metadata.get("tempo", 0)
        velocity = metadata.get("avg_velocity", 0)

        if tempo < 80:
            tempo_str = "slow tempo"
        elif tempo < 140:
            tempo_str = "moderate tempo"
        else:
            tempo_str = "fast tempo"
        
        if velocity < 50:
            velocity_str = "soft dynamics"
        elif velocity < 85:
            velocity_str = "moderate dynamics"
        else:
            velocity_str = "loud dynamics"
        
        return f"piano piece in {key}, {tempo_str} at {tempo:.0f} BPM, {velocity_str}"

    def build_index_from_dataset(self, dataset: MIDIDataset, max_files: int = None) -> None:
        """
        Convenience method: builds the index directly from a MIDIDataset.
        Extracts metadata from each file and auto-generates descriptions.
        Args:
            dataset: A MIDIDataset instance with files loaded.
            max_files: Optional cap on how many files to index (useful for testing).
        """
        files = dataset.files
        if max_files:
            files = files[:max_files]

        snippets = []
        for i, midi_path in enumerate(files):
            midi_path_str = str(midi_path)
            try:
                metadata = dataset.extract_metadata(midi_path_str)
                description = self._make_description(midi_path_str, metadata)

                # Infer basic mood tags from metadata
                mood_tags = []
                if metadata["tempo"] < 80:
                    mood_tags.append("slow")
                elif metadata["tempo"] > 140:
                    mood_tags.append("fast")
                else:
                    mood_tags.append("moderate")

                if metadata["avg_velocity"] < 50:
                    mood_tags.extend(["soft", "gentle", "melancholic"])
                elif metadata["avg_velocity"] > 85:
                    mood_tags.extend(["loud", "dramatic", "powerful"])
                else:
                    mood_tags.extend(["balanced", "expressive"])

                snippets.append(MIDISnippet(
                    file_path=midi_path_str,
                    description=description,
                    mood_tags=mood_tags,
                    metadata=metadata
                ))

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(files)} files...")

            except Exception as e:
                print(f"  Skipping {midi_path_str}: {e}")
                continue

        self.build_index(snippets)

    def build_index(self, snippets: list[MIDISnippet]) -> None:
        """Embeds all snippet descriptions and builds the FAISS index."""
        self.snippets = snippets
        descriptions = [s.description for s in snippets]

        print(f"Embedding {len(descriptions)} descriptions...")
        embeddings = self.model.encode(descriptions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def save_index(self, path: str) -> None:
        """Saves the FAISS index and snippet metadata to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        faiss.write_index(self.index, path + ".faiss")

        with open(path + ".snippets.pkl", "wb") as f:
            pickle.dump(self.snippets, f)

        print(f"Index saved to {path}.faiss and {path}.snippets.pkl")

    def load_index(self, path: str) -> None:
        """Loads a pre-built FAISS index from disk."""
        faiss_path = path if path.endswith(".faiss") else path + ".faiss"
        snippets_path = faiss_path.replace(".faiss", ".snippets.pkl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not os.path.exists(snippets_path):
            raise FileNotFoundError(f"Snippets file not found at {snippets_path}")

        self.index = faiss.read_index(faiss_path)

        with open(snippets_path, "rb") as f:
            self.snippets = pickle.load(f)

        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.snippets)} snippets.")

    def retrieve(self, query: str, top_k: int = 3) -> list[MIDISnippet]:
        """
        Given a natural language mood/style query, returns the top_k most
        relevant MIDI snippets.
        Args:
            query: e.g. "melancholic, slow, minor key"
            top_k: Number of results to return.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Run build_index() first.")

        query_embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            snippet = self.snippets[idx]
            results.append(snippet)

        return results

    