import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.midi_dataset import MIDIDataset


@dataclass
class MIDISnippet:
    file_path: str
    description: str
    mood_tags: list[str]
    metadata: dict
    retrieval_text: str


class RAGRetriever:
    """
    Builds and queries a FAISS vector index of MIDI snippet embeddings.
    Used to retrieve stylistically relevant MIDI files/snippets given a mood/style query.
    """

    def __init__(self, index_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.embedding_model_name = embedding_model

        print("Loading embedding model...")
        self.model = SentenceTransformer(embedding_model)
        print("Model loaded.")

        self.index = None
        self.snippets: list[MIDISnippet] = []

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def _tempo_bucket(self, tempo: float) -> str:
        """
        Buckets tuned to observed pretty_midi tempo estimates on MAESTRO.
        """
        if tempo < 110:
            return "slow"
        elif tempo < 190:
            return "moderate"
        return "fast"

    def _velocity_bucket(self, velocity: float) -> str:
        if velocity < 45:
            return "soft"
        elif velocity < 80:
            return "moderate"
        return "loud"

    def _make_description(self, metadata: dict) -> str:
        key = metadata.get("key", "Unknown")
        mode = metadata.get("mode", "unknown")
        tempo = metadata.get("tempo", 0.0)
        velocity = metadata.get("avg_velocity", 0.0)

        tempo_label = self._tempo_bucket(tempo)
        velocity_label = self._velocity_bucket(velocity)

        key_text = f"{key} {mode}".strip() if key != "Unknown" else "unknown key"

        return (
            f"classical solo piano piece, {tempo_label} tempo, "
            f"{key_text}, {velocity_label} dynamics, approximately {tempo:.0f} BPM"
        )

    def _make_tags(self, metadata: dict) -> list[str]:
        tempo = metadata.get("tempo", 0.0)
        velocity = metadata.get("avg_velocity", 0.0)
        mode = metadata.get("mode", "unknown")

        tempo_label = self._tempo_bucket(tempo)
        dyn_label = self._velocity_bucket(velocity)

        tags = {tempo_label, dyn_label, mode, "expressive"}

        if mode == "minor":
            tags.update(["dark", "somber"])
        elif mode == "major":
            tags.update(["bright"])

        if dyn_label == "soft" and mode == "minor":
            tags.update(["melancholic", "gentle", "lyrical", "nocturne-like"])
        elif dyn_label == "soft" and mode == "major":
            tags.update(["gentle", "tender", "calm"])
        elif dyn_label == "loud" and tempo_label == "fast":
            tags.update(["dramatic", "powerful", "agitated"])
        elif dyn_label == "moderate" and mode == "minor":
            tags.update(["reflective", "emotional"])

        return sorted(tags)

    def _make_retrieval_text(self, description: str, mood_tags: list[str], metadata: dict) -> str:
        return (
            f"{description}. "
            f"Tags: {', '.join(mood_tags)}. "
            f"Style hints: classical piano, romantic piano, expressive solo piano, "
            f"lyrical piano, recital piano."
        )

    def build_index_from_dataset(self, dataset: MIDIDataset, max_files: int = None) -> None:
        """
        Builds the index directly from a MIDIDataset.
        Uses randomized sampling for better diversity when max_files is provided.
        """
        files = list(dataset.files)
        if max_files is not None:
            random.seed(42)
            files = random.sample(files, min(max_files, len(files)))

        snippets = []
        for i, midi_path in enumerate(files):
            midi_path_str = str(midi_path)
            try:
                metadata = dataset.extract_metadata(midi_path_str)
                description = self._make_description(metadata)
                mood_tags = self._make_tags(metadata)
                retrieval_text = self._make_retrieval_text(description, mood_tags, metadata)

                snippets.append(
                    MIDISnippet(
                        file_path=midi_path_str,
                        description=description,
                        mood_tags=mood_tags,
                        metadata=metadata,
                        retrieval_text=retrieval_text,
                    )
                )

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(files)} files...")

            except Exception as e:
                print(f"  Skipping {midi_path_str}: {e}")

        self.build_index(snippets)

    def build_index(self, snippets: list[MIDISnippet]) -> None:
        """Embeds all retrieval texts and builds the FAISS index."""
        if not snippets:
            raise ValueError("No snippets provided to build_index().")

        self.snippets = snippets
        texts = [s.retrieval_text for s in snippets]

        print(f"Embedding {len(texts)} descriptions...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
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

    def _extract_query_preferences(self, query: str) -> dict:
        q = query.lower()

        prefs = {
            "tempo": None,
            "mode": None,
            "soft": False,
            "loud": False,
            "gentle": False,
            "dramatic": False,
            "melancholic": False,
            "nocturne": False,
        }

        if "slow" in q:
            prefs["tempo"] = "slow"
        elif "moderate" in q:
            prefs["tempo"] = "moderate"
        elif "fast" in q:
            prefs["tempo"] = "fast"

        if "minor" in q:
            prefs["mode"] = "minor"
        elif "major" in q:
            prefs["mode"] = "major"

        if "soft" in q:
            prefs["soft"] = True
        if "loud" in q or "powerful" in q:
            prefs["loud"] = True
        if "gentle" in q:
            prefs["gentle"] = True
        if "dramatic" in q:
            prefs["dramatic"] = True
        if "melancholic" in q or "sad" in q:
            prefs["melancholic"] = True
        if "nocturne" in q:
            prefs["nocturne"] = True

        return prefs

    def _passes_constraint_filter(self, prefs: dict, snippet: MIDISnippet) -> bool:
        """
        Lightweight symbolic filter.
        For explicit tempo/mode constraints, reject strong contradictions.
        """
        tags = set(snippet.mood_tags)
        mode = snippet.metadata.get("mode", "unknown")

        if prefs["tempo"] == "slow" and "fast" in tags:
            return False
        if prefs["tempo"] == "fast" and "slow" in tags:
            return False
        if prefs["mode"] == "minor" and mode == "major":
            return False
        if prefs["mode"] == "major" and mode == "minor":
            return False

        return True

    def _rerank(self, query: str, candidates: list[Tuple[float, MIDISnippet]]) -> list[MIDISnippet]:
        prefs = self._extract_query_preferences(query)
        reranked = []

        for base_score, snippet in candidates:
            score = float(base_score)
            tags = set(snippet.mood_tags)
            mode = snippet.metadata.get("mode", "unknown")

            # Tempo handling: reward matches, penalize contradictions
            if prefs["tempo"] == "slow":
                if "slow" in tags:
                    score += 0.30
                elif "moderate" in tags:
                    score -= 0.08
                elif "fast" in tags:
                    score -= 0.40

            elif prefs["tempo"] == "moderate":
                if "moderate" in tags:
                    score += 0.20
                else:
                    score -= 0.05

            elif prefs["tempo"] == "fast":
                if "fast" in tags:
                    score += 0.30
                elif "moderate" in tags:
                    score -= 0.08
                elif "slow" in tags:
                    score -= 0.40

            # Mode handling
            if prefs["mode"] == "minor":
                if mode == "minor":
                    score += 0.22
                elif mode == "major":
                    score -= 0.22

            elif prefs["mode"] == "major":
                if mode == "major":
                    score += 0.22
                elif mode == "minor":
                    score -= 0.22

            # Other preferences
            if prefs["soft"]:
                if "soft" in tags:
                    score += 0.14
                elif "loud" in tags:
                    score -= 0.14

            if prefs["loud"]:
                if "loud" in tags:
                    score += 0.14
                elif "soft" in tags:
                    score -= 0.14

            if prefs["gentle"]:
                if "gentle" in tags or "tender" in tags or "calm" in tags:
                    score += 0.14
                if "agitated" in tags:
                    score -= 0.12

            if prefs["dramatic"]:
                if "dramatic" in tags or "powerful" in tags or "agitated" in tags:
                    score += 0.16
                if "gentle" in tags:
                    score -= 0.10

            if prefs["melancholic"]:
                if "melancholic" in tags or "somber" in tags or "dark" in tags:
                    score += 0.16
                if "bright" in tags:
                    score -= 0.10

            if prefs["nocturne"]:
                if "nocturne-like" in tags or "lyrical" in tags:
                    score += 0.16

            reranked.append((score, snippet))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [snippet for _, snippet in reranked]

    def retrieve(self, query: str, top_k: int = 3) -> list[MIDISnippet]:
        """
        Returns the top_k most relevant MIDI snippets for a natural language query.
        Uses semantic retrieval first, then symbolic filtering/reranking.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Run build_index() first.")

        query_embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        fetch_k = min(max(top_k * 10, 20), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            candidates.append((score, self.snippets[idx]))

        prefs = self._extract_query_preferences(query)

        # Try constraint-aware filtering first
        filtered = [
            (score, snippet)
            for score, snippet in candidates
            if self._passes_constraint_filter(prefs, snippet)
        ]

        # If filtering becomes too strict, fall back to all candidates
        active_candidates = filtered if len(filtered) >= top_k else candidates

        reranked = self._rerank(query, active_candidates)
        return reranked[:top_k]