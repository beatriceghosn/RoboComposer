from dataclasses import dataclass
from typing import List
import pretty_midi


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for MIDI event quantization."""

    time_step_seconds: float = 0.01
    max_time_shift_steps: int = 100
    velocity_bins: int = 32
    max_duration_steps: int = 100
    add_bos_eos: bool = True


class MIDITokenizer:
    """
    Converts MIDI notes to a integer token sequence.

    Event layout:
    - TimeShift token(s)
    - NoteOn token (pitch)
    - Velocity token 
    - Duration token
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()

        self.time_shift_base = 4
        self.note_on_base = self.time_shift_base + self.config.max_time_shift_steps
        self.velocity_base = self.note_on_base + 128
        self.duration_base = self.velocity_base + self.config.velocity_bins
        self.vocab_size = self.duration_base + self.config.max_duration_steps

    def tokenize(self, midi_path: str) -> List[int]:
        """Tokenize all notes from a MIDI file into integer IDs."""
        midi = pretty_midi.PrettyMIDI(midi_path)
        return self.tokenize_midi(midi)

    def tokenize_midi(self, midi: pretty_midi.PrettyMIDI) -> List[int]:
        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            notes.extend(instrument.notes)

        notes.sort(key=lambda n: (n.start, n.pitch, n.end))

        tokens: List[int] = []
        if self.config.add_bos_eos:
            tokens.append(self.BOS)

        current_time = 0.0
        for note in notes:
            delta = max(0.0, note.start - current_time)
            tokens.extend(self._encode_time_delta(delta))
            tokens.append(self._encode_pitch(note.pitch))
            tokens.append(self._encode_velocity(note.velocity))
            tokens.append(self._encode_duration(note.end - note.start))
            current_time = note.start

        if self.config.add_bos_eos:
            tokens.append(self.EOS)

        return tokens

    def _encode_time_delta(self, delta_seconds: float) -> List[int]:
        step = self.config.time_step_seconds
        max_steps = self.config.max_time_shift_steps
        steps = int(round(delta_seconds / step))
        if steps <= 0:
            return []

        out: List[int] = []
        while steps > 0:
            chunk = min(steps, max_steps)
            out.append(self.time_shift_base + chunk - 1)
            steps -= chunk
        return out

    def _encode_pitch(self, pitch: int) -> int:
        pitch = max(0, min(127, int(pitch)))
        return self.note_on_base + pitch

    def _encode_velocity(self, velocity: int) -> int:
        velocity = max(1, min(127, int(velocity)))
        bin_size = 127 / self.config.velocity_bins
        velocity_bin = min(
            self.config.velocity_bins - 1, int((velocity - 1) / bin_size)
        )
        return self.velocity_base + velocity_bin

    def _encode_duration(self, duration_seconds: float) -> int:
        steps = max(1, int(round(duration_seconds / self.config.time_step_seconds)))
        steps = min(self.config.max_duration_steps, steps)
        return self.duration_base + steps - 1


def tokenize_midi_file(midi_path: str, config: TokenizerConfig | None = None) -> List[int]:
    """Convenience function for one-off tokenization."""
    return MIDITokenizer(config=config).tokenize(midi_path)
