import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pretty_midi
from src.tokenizer import MIDITokenizer

class MIDIDataset(Dataset):
    """
    Loads and tokenizes MIDI files from the MAESTRO dataset.
    Converts raw MIDI into token sequences suitable for transformer input.
    """

    def __init__(self, raw_dir="data/maestro-v3.0.0", processed_dir = "data/processed", split = "train", max_seq_len = 1024):
        self.split = split
        self.max_seq_len = max_seq_len

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {self.raw_dir}")

        #  Collect all MIDI files across year folders
        self.files = sorted(
            list(self.raw_dir.glob("*/*.mid")) +
            list(self.raw_dir.glob("*/*.midi"))
        )
        # self.files = sorted(
        #     list((self.raw_dir / "2004").glob("*.mid")) +
        #     list((self.raw_dir / "2004").glob("*.midi"))
        # )

        if len(self.files) == 0:
            print(f"No MIDI files found in {self.raw_dir}")

        self.tokenizer = MIDITokenizer()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """Returns a tokenized tensor for one MIDI file."""
        item = self.files[idx]
        midi_path = item

        tokens = self.tokenize(str(midi_path))

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens += [0] * (self.max_seq_len - len(tokens))

        return torch.tensor(tokens)

    def tokenize(self, midi_path: str) -> list[int]:
        """Converts a MIDI file to a list of integer tokens."""
        return self.tokenizer.tokenize(midi_path)

    def extract_metadata(self, midi_path: str) -> dict:
        """
        Extracts tempo, key, and average velocity from a MIDI file.
        Returns: {"tempo": float, "key": str, "avg_velocity": float}
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        tempo = float(midi.estimate_tempo())

        # MAESTRO files are usually single-key passages; this keeps key extraction lightweight.
        chroma = midi.get_chroma()
        if chroma.size == 0 or chroma.sum() == 0:
            key_name = "Unknown"
        else:
            pitch_class = int(chroma.sum(axis=1).argmax())
            major_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key_name = major_names[pitch_class]

        velocities = [
            note.velocity
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ]
        avg_velocity = float(sum(velocities) / len(velocities)) if velocities else 0.0

        return {"tempo": tempo, "key": key_name, "avg_velocity": avg_velocity}