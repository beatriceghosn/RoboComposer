import os
from torch.utils.data import Dataset

class MIDIDataset(Dataset):
    """
    Loads and tokenizes MIDI files from the MAESTRO dataset.
    Converts raw MIDI into token sequences suitable for transformer input.
    """

    def __init__(self, data_dir: str, split: str = "train", max_seq_len: int = 1024):
        """
        Args:
            data_dir: Path to the MAESTRO dataset root.
            split: One of "train", "validation", "test".
            max_seq_len: Maximum token sequence length.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.files = []  # TODO: populate with MIDI file paths for this split
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """Returns a tokenized tensor for one MIDI file."""
        raise NotImplementedError

    def tokenize(self, midi_path: str) -> list[int]:
        """Converts a MIDI file to a list of integer tokens using miditok."""
        raise NotImplementedError

    def extract_metadata(self, midi_path: str) -> dict:
        """
        Extracts tempo, key, and average velocity from a MIDI file.
        Returns: {"tempo": float, "key": str, "avg_velocity": float}
        """
        raise NotImplementedError