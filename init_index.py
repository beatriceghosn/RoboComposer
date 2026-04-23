import os
import sys
from src.midi_dataset import MIDIDataset
from src.rag_retriever import RAGRetriever

def run():
    # Setup paths
    output_dir = "data/processed"
    index_name = "rag_index"
    save_path = os.path.join(output_dir, index_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    print(f"Reading 'train' split from data/raw...")
    try:
        dataset = MIDIDataset(split="train")
    except Exception as e:
        print(f"Fatal: Could not load dataset. {e}")
        sys.exit(1)

    if not dataset.files:
        print("No MIDI files detected. Check your data/raw directory structure.")
        sys.exit(1)

    # 2. Build Index
    print(f"Initializing RAGRetriever with {len(dataset.files)} files...")
    retriever = RAGRetriever()
    
    try:
        retriever.build_index_from_dataset(dataset)
    except Exception as e:
        print(f"Indexing failed: {e}")
        sys.exit(1)

    # 3. Save
    print(f"Saving artifacts to {output_dir}...")
    retriever.save_index(save_path)
    
    print("Done. Indexing complete.")

if __name__ == "__main__":
    run()