from dataclasses import dataclass

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
        """
        Args:
            index_path: Path to a pre-built FAISS index file. If None, starts fresh.
            embedding_model: SentenceTransformer model name for embedding descriptions.
        """
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self.index = None       # FAISS index
        self.snippets = []      # list of MIDISnippet, parallel to index rows
        raise NotImplementedError

    def build_index(self, snippets: list[MIDISnippet]) -> None:
        """Embeds all snippet descriptions and builds the FAISS index."""
        raise NotImplementedError

    def save_index(self, path: str) -> None:
        """Saves the FAISS index and snippet metadata to disk."""
        raise NotImplementedError

    def load_index(self, path: str) -> None:
        """Loads a pre-built FAISS index from disk."""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 3) -> list[MIDISnippet]:
        """
        Given a natural language mood/style query, returns the top_k most
        relevant MIDI snippets.
        Args:
            query: e.g. "melancholic, slow, minor key"
            top_k: Number of results to return.
        """
        raise NotImplementedError