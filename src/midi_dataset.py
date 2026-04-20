import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pretty_midi
from src.tokenizer import MIDITokenizer
import numpy as np

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
        Extract tempo, key, mode, and average velocity from a MIDI file.
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        # more stable than relying only on estimate_tempo()
        tempo_times, tempi = midi.get_tempo_changes()
        if len(tempi) == 0:
            tempo = float(midi.estimate_tempo())
        else:
            tempo = float(np.median(tempi))

        chroma = midi.get_chroma()
        if chroma.size == 0 or chroma.sum() == 0:
            key_name = "Unknown"
            mode = "unknown"
        else:
            chroma_sum = chroma.sum(axis=1)

            major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                    2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

            best_score = -1e9
            best_key = "C"
            best_mode = "major"

            for shift in range(12):
                maj_score = np.dot(chroma_sum, np.roll(major_template, shift))
                min_score = np.dot(chroma_sum, np.roll(minor_template, shift))

                if maj_score > best_score:
                    best_score = maj_score
                    best_key = names[shift]
                    best_mode = "major"

                if min_score > best_score:
                    best_score = min_score
                    best_key = names[shift]
                    best_mode = "minor"

            key_name = best_key
            mode = best_mode

        velocities = [
            note.velocity
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ]
        avg_velocity = float(sum(velocities) / len(velocities)) if velocities else 0.0

        return {
            "tempo": tempo,
            "key": key_name,
            "mode": mode,
            "avg_velocity": avg_velocity
        }