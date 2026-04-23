import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.music_generator import MusicGenerator
from src.rag_retriever import RAGRetriever


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Initializing MusicGenerator ---")
    generator = MusicGenerator()

    # Test 1: Generate without RAG context
    constraints_1 = {
        "mood": "melancholic",
        "tempo": "slow",
        "key": "D minor",
        "style": "Chopin",
        "notes": "gentle nocturne with left hand arpeggios"
    }

    print("\n--- Test 1: Generating without RAG context ---")
    midi_bytes_1 = generator.generate(
        constraints=constraints_1,
        context_snippets=None,
        max_tokens=512,
        temperature=0.9,
    )

    assert isinstance(midi_bytes_1, bytes), "Generated output is not bytes."
    assert len(midi_bytes_1) > 0, "Generated MIDI bytes are empty."

    output_path_1 = output_dir / "test_no_rag.mid"
    generator.save_midi(midi_bytes_1, str(output_path_1))

    assert output_path_1.exists(), "MIDI file was not saved."
    assert output_path_1.stat().st_size > 0, "Saved MIDI file is empty."

    print(f"Saved: {output_path_1}")
    print(f"Size: {output_path_1.stat().st_size} bytes")

    # Test 2: Generate with RAG context
    print("\n--- Test 2: Generating with RAG context ---")

    rag_index_path = "data/processed/rag_index.faiss"
    context_snippets = None

    if os.path.exists(rag_index_path):
        retriever = RAGRetriever(index_path=rag_index_path)
        context_snippets = retriever.retrieve(
            "melancholic slow gentle nocturne in D minor",
            top_k=3
        )
        print(f"Retrieved {len(context_snippets)} context snippets.")
        for i, snippet in enumerate(context_snippets):
            print(f"  {i + 1}. {snippet.description}")
    else:
        print("No saved RAG index found. Skipping context retrieval for Test 2.")

    constraints_2 = {
        "mood": "gentle",
        "tempo": "slow",
        "key": "D minor",
        "style": "Debussy",
        "notes": "soft expressive nocturne with flowing arpeggios"
    }

    midi_bytes_2 = generator.generate(
        constraints=constraints_2,
        context_snippets=context_snippets,
        max_tokens=640,
        temperature=0.85,
    )

    assert isinstance(midi_bytes_2, bytes), "Generated output with context is not bytes."
    assert len(midi_bytes_2) > 0, "Generated MIDI bytes with context are empty."

    output_path_2 = output_dir / "test_with_rag.mid"
    generator.save_midi(midi_bytes_2, str(output_path_2))

    assert output_path_2.exists(), "Context-based MIDI file was not saved."
    assert output_path_2.stat().st_size > 0, "Saved context-based MIDI file is empty."

    print(f"Saved: {output_path_2}")
    print(f"Size: {output_path_2.stat().st_size} bytes")

    # Test 3: Different mood/style sanity check
    
    print("\n--- Test 3: Fast dramatic sanity check ---")

    constraints_3 = {
        "mood": "dramatic",
        "tempo": "fast",
        "key": "C minor",
        "style": "Beethoven",
        "notes": "powerful accented piano writing"
    }

    midi_bytes_3 = generator.generate(
        constraints=constraints_3,
        context_snippets=None,
        max_tokens=512,
        temperature=1.0,
    )

    assert isinstance(midi_bytes_3, bytes), "Generated dramatic output is not bytes."
    assert len(midi_bytes_3) > 0, "Generated dramatic MIDI bytes are empty."

    output_path_3 = output_dir / "test_dramatic.mid"
    generator.save_midi(midi_bytes_3, str(output_path_3))

    assert output_path_3.exists(), "Dramatic MIDI file was not saved."
    assert output_path_3.stat().st_size > 0, "Saved dramatic MIDI file is empty."

    print(f"Saved: {output_path_3}")
    print(f"Size: {output_path_3.stat().st_size} bytes")

    print("\nAll music generator tests passed.")


if __name__ == "__main__":
    main()