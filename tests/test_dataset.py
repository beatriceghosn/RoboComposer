import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.tokenizer import MIDITokenizer
from src.midi_dataset import MIDIDataset

# Auto-find a real file from your MAESTRO folder
midi_path = str(sorted(Path("data/maestro-v3.0.0").glob("*/*.midi"))[0])
print(f"Using file: {midi_path}\n")

# Test the tokenizer on a single file
print("--- Testing tokenizer ---")
tokenizer = MIDITokenizer()
tokens = tokenizer.tokenize(midi_path)
print(f"Token count: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Test the dataset
print("\n--- Testing MIDIDataset ---")
dataset = MIDIDataset(raw_dir="data/maestro-v3.0.0")
print(f"Files found: {len(dataset)}")

sample = dataset[0]
print(f"Sample tensor shape: {sample.shape}")
print(f"Sample dtype: {sample.dtype}")

# Test metadata extraction
print("\n--- Testing metadata extraction ---")
meta = dataset.extract_metadata(midi_path)
print(f"Tempo: {meta['tempo']:.1f} BPM")
print(f"Key: {meta['key']}")
print(f"Avg velocity: {meta['avg_velocity']:.1f}")