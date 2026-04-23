import os
from pathlib import Path

from src.llm_orchestrator import LLMOrchestrator
from src.rag_retriever import RAGRetriever
from src.music_generator import MusicGenerator


def main():
    prompt = "Compose a gentle melancholic nocturne in D minor, slow tempo, in the style of Chopin, with flowing left hand arpeggios."

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: Parse user prompt with LLMOrchestrator ---")
    orchestrator = LLMOrchestrator()
    constraints = orchestrator.parse_user_prompt(prompt)
    print("Parsed constraints:")
    print(constraints)

    print("\n--- Step 2: Retrieve RAG context ---")
    retriever = RAGRetriever(index_path="data/processed/rag_index.faiss")

    rag_query = " ".join([
        constraints.get("mood", ""),
        constraints.get("tempo", ""),
        constraints.get("key", ""),
        constraints.get("style", ""),
        constraints.get("notes", "")
    ]).strip()

    context_snippets = retriever.retrieve(rag_query, top_k=3)

    print(f"Retrieved {len(context_snippets)} snippets:")
    for i, snippet in enumerate(context_snippets, start=1):
        print(f"{i}. {snippet.description}")
        print(f"   tags: {snippet.mood_tags}")
        print(f"   file: {os.path.basename(snippet.file_path)}")

    print("\n--- Step 3: Generate MIDI ---")
    generator = MusicGenerator()
    midi_bytes = generator.generate(
        constraints=constraints,
        context_snippets=context_snippets,
        max_tokens=512,
        temperature=0.9,
    )

    output_path = output_dir / "full_pipeline_output.mid"
    generator.save_midi(midi_bytes, str(output_path))

    print("\n--- Step 4: Done ---")
    print(f"Saved MIDI to: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()