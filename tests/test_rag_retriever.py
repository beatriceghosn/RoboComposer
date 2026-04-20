import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.midi_dataset import MIDIDataset
from src.rag_retriever import RAGRetriever


def main():
    print("--- Building index from dataset (random 300 files) ---\n")
    dataset = MIDIDataset(raw_dir="data/maestro-v3.0.0")

    print("--- Metadata sanity check ---")
    for midi_path in dataset.files[:10]:
        md = dataset.extract_metadata(str(midi_path))
        print(os.path.basename(str(midi_path)), md)

    retriever = RAGRetriever()
    retriever.build_index_from_dataset(dataset, max_files=300)

    print("\n--- Saving index ---")
    retriever.save_index("data/processed/rag_index")

    print("\n--- Testing retrieval ---\n")
    queries = [
        "melancholic slow piece in a minor key",
        "fast dramatic powerful Beethoven style",
        "gentle soft expressive nocturne",
        "bright calm major key piano piece",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, snippet in enumerate(results):
            print(f"  {i + 1}. {snippet.description}")
            print(f"     tags: {snippet.mood_tags}")
            print(f"     file: {os.path.basename(snippet.file_path)}")
        print()

    print("--- Loading saved index and re-querying ---\n")
    retriever2 = RAGRetriever(index_path="data/processed/rag_index.faiss")
    results = retriever2.retrieve("slow melancholic minor nocturne", top_k=2)
    for snippet in results:
        print(f"  {snippet.description}")


if __name__ == "__main__":
    main()