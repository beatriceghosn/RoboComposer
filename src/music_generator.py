import io
import os
import math
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi


class MusicGenerator:
    """
    Baseline MIDI generator for RoboComposer.
    - Uses structured constraints from LLMOrchestrator
    - Optionally extracts pitch/rhythm/style priors from retrieved MIDI snippets
    - Generates a new PrettyMIDI object
    - Returns raw MIDI bytes
    """

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
    MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

    STYLE_PRESETS = {
        "chopin": {"register": (55, 84), "polyphony": 0.35, "rubato": 0.035},
        "debussy": {"register": (50, 86), "polyphony": 0.45, "rubato": 0.040},
        "baroque": {"register": (52, 79), "polyphony": 0.20, "rubato": 0.010},
        "beethoven": {"register": (48, 88), "polyphony": 0.40, "rubato": 0.020},
        "romantic": {"register": (50, 85), "polyphony": 0.40, "rubato": 0.030},
        "classical": {"register": (52, 84), "polyphony": 0.25, "rubato": 0.015},
    }

    MOOD_PRESETS = {
        "melancholic": {"velocity_shift": -8, "density": 0.85, "legato": 0.90},
        "joyful": {"velocity_shift": 6, "density": 1.10, "legato": 0.70},
        "tense": {"velocity_shift": 8, "density": 1.15, "legato": 0.60},
        "gentle": {"velocity_shift": -12, "density": 0.80, "legato": 0.95},
        "dramatic": {"velocity_shift": 12, "density": 1.20, "legato": 0.70},
        "calm": {"velocity_shift": -10, "density": 0.75, "legato": 0.92},
        "mysterious": {"velocity_shift": -2, "density": 0.90, "legato": 0.85},
    }

    def __init__(self, model_path: str = None, device: str = "cpu", seed: Optional[int] = 42):
        """
        Args:
            model_path: Reserved for future fine-tuned model weights.
            device: "cpu" or "cuda".
            seed: Optional random seed for reproducible generation.
        """
        self.model_path = model_path
        self.device = device
        self.model = None  # future learned model hook

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Placeholder model loading hook for future transformer-based generation
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            # For now we do not load an ML model yet.
            # This keeps the class usable today while preserving the API.
            self.model = {"model_path": model_path, "device": device}

    def generate(
        self,
        constraints: dict,
        context_snippets: list = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
    ) -> bytes:
        """
        Generates MIDI bytes from structured constraints and optional RAG context.

        Args:
            constraints: dict like:
                {
                    "mood": "melancholic",
                    "tempo": "slow",
                    "key": "D minor",
                    "style": "Chopin",
                    "notes": "gentle left hand arpeggios"
                }
            context_snippets: Optional list of retrieved snippet objects from RAGRetriever
            max_tokens: Used as a proxy for desired piece length
            temperature: Controls randomness in note selection

        Returns:
            Raw MIDI bytes.
        """
        midi = self._generate_midi(constraints, context_snippets, max_tokens, temperature)
        return self._midi_to_bytes(midi)

    def save_midi(self, midi_bytes: bytes, output_path: str) -> None:
        """Writes generated MIDI bytes to a .mid file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(midi_bytes)

    def _generate_midi(
        self,
        constraints: dict,
        context_snippets: list = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
    ) -> pretty_midi.PrettyMIDI:
        mood = str(constraints.get("mood", "expressive")).strip().lower()
        tempo_label = str(constraints.get("tempo", "moderate")).strip().lower()
        key_str = str(constraints.get("key", "C major")).strip()
        style = str(constraints.get("style", "classical")).strip().lower()
        notes_text = str(constraints.get("notes", "")).strip().lower()

        root_name, mode = self._parse_key(key_str)
        bpm = self._resolve_bpm(tempo_label, mood, style, notes_text)

        style_cfg = self._get_style_preset(style)
        mood_cfg = self._get_mood_preset(mood)

        context_profile = self._extract_context_profile(context_snippets or [], root_name, mode)

        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

        beats_per_bar = 4
        beat_seconds = 60.0 / bpm

        # approximate piece length from max_tokens
        approx_notes = max(24, min(140, max_tokens // 4))
        bars = max(8, min(32, approx_notes // 4))

        register_low, register_high = style_cfg["register"]
        legato = mood_cfg["legato"]
        polyphony_prob = style_cfg["polyphony"]
        density = mood_cfg["density"]
        rubato_amt = style_cfg["rubato"]

        scale_pitches = self._build_scale_pitches(root_name, mode, register_low, register_high)

        melody_notes = self._compose_melody(
            bpm=bpm,
            bars=bars,
            beat_seconds=beat_seconds,
            scale_pitches=scale_pitches,
            context_profile=context_profile,
            mood=mood,
            style=style,
            density=density,
            legato=legato,
            rubato_amt=rubato_amt,
            temperature=temperature,
            register_low=register_low,
            register_high=register_high,
        )

        accompaniment_notes = self._compose_accompaniment(
            bpm=bpm,
            bars=bars,
            beat_seconds=beat_seconds,
            root_name=root_name,
            mode=mode,
            mood=mood,
            style=style,
            notes_text=notes_text,
            context_profile=context_profile,
            density=density,
            legato=legato,
            rubato_amt=rubato_amt,
            polyphony_prob=polyphony_prob,
        )

        all_notes = melody_notes + accompaniment_notes
        all_notes.sort(key=lambda n: (n.start, n.pitch))

        for note in all_notes:
            piano.notes.append(note)

        midi.instruments.append(piano)
        return midi

    def _parse_key(self, key_str: str) -> tuple[str, str]:
        key_str = key_str.strip()
        lowered = key_str.lower()

        mode = "major"
        if "minor" in lowered:
            mode = "minor"
        elif "major" in lowered:
            mode = "major"

        parts = key_str.split()
        root = parts[0].upper() if parts else "C"

        # normalize note spelling
        replacements = {
            "DB": "C#",
            "EB": "D#",
            "GB": "F#",
            "AB": "G#",
            "BB": "A#",
        }
        root = replacements.get(root, root)

        if root not in self.NOTE_NAMES:
            root = "C"

        return root, mode

    def _resolve_bpm(self, tempo_label: str, mood: str, style: str, notes_text: str) -> float:
        base = {
            "slow": 76,
            "moderate": 108,
            "fast": 144,
        }.get(tempo_label, 108)

        if mood in {"melancholic", "gentle", "calm"}:
            base -= 8
        elif mood in {"dramatic", "tense", "joyful"}:
            base += 8

        if "nocturne" in notes_text:
            base -= 10
        if "waltz" in notes_text:
            base += 6
        if "chase" in notes_text:
            base += 16

        if style == "debussy":
            base -= 4
        elif style == "beethoven":
            base += 6

        return float(max(50, min(180, base)))

    def _get_style_preset(self, style: str) -> dict:
        style = style.lower()
        for name, preset in self.STYLE_PRESETS.items():
            if name in style:
                return preset
        return self.STYLE_PRESETS["classical"]

    def _get_mood_preset(self, mood: str) -> dict:
        mood = mood.lower()
        for name, preset in self.MOOD_PRESETS.items():
            if name in mood:
                return preset
        return {"velocity_shift": 0, "density": 1.0, "legato": 0.82}


    def _extract_context_profile(self, snippets: list, root_name: str, mode: str) -> dict:
        """
        Extract simple musical priors from retrieved MIDI files:
        - pitch class histogram
        - register preference
        - duration preference
        - velocity preference
        """
        pitch_hist = np.ones(12, dtype=np.float64)
        note_pitches = []
        durations = []
        velocities = []

        for snippet in snippets:
            midi_path = getattr(snippet, "file_path", None)
            if not midi_path or not os.path.exists(midi_path):
                continue

            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
                for instrument in midi.instruments:
                    if instrument.is_drum:
                        continue
                    for note in instrument.notes:
                        pitch_hist[note.pitch % 12] += 1.0
                        note_pitches.append(note.pitch)
                        durations.append(max(0.05, note.end - note.start))
                        velocities.append(note.velocity)
            except Exception:
                continue

        pitch_hist = pitch_hist / pitch_hist.sum()

        if note_pitches:
            register_center = float(np.median(note_pitches))
        else:
            register_center = 67.0  # around G4

        if durations:
            mean_duration = float(np.median(durations))
        else:
            mean_duration = 0.5

        if velocities:
            mean_velocity = float(np.median(velocities))
        else:
            mean_velocity = 64.0

        # bias toward requested key as well
        root_pc = self.NOTE_NAMES.index(root_name)
        allowed_scale = self.MINOR_INTERVALS if mode == "minor" else self.MAJOR_INTERVALS
        for interval in allowed_scale:
            pitch_hist[(root_pc + interval) % 12] += 0.5
        pitch_hist = pitch_hist / pitch_hist.sum()

        return {
            "pitch_hist": pitch_hist,
            "register_center": register_center,
            "mean_duration": mean_duration,
            "mean_velocity": mean_velocity,
        }


    def _compose_melody(
    self,
    bpm: float,
    bars: int,
    beat_seconds: float,
    scale_pitches: list[int],
    context_profile: dict,
    mood: str,
    style: str,
    density: float,
    legato: float,
    rubato_amt: float,
    temperature: float,
    register_low: int,
    register_high: int,
) -> list[pretty_midi.Note]:

        notes = []
        t = 0.0

        target_events = int(bars * 4 * density)
        target_events = max(24, min(128, target_events))

        prev_pitch = None
        pitch_weights = self._pitch_weights_for_scale(scale_pitches, context_profile)

        for _ in range(target_events):
            # --- duration ---
            dur_beats = self._sample_duration_beats(mood, style)
            dur_seconds = dur_beats * beat_seconds

            pitch = self._sample_pitch(
                scale_pitches=scale_pitches,
                pitch_weights=pitch_weights,
                prev_pitch=prev_pitch,
                register_low=register_low,
                register_high=register_high,
                temperature=temperature,
            )

            velocity = self._sample_velocity(mood, context_profile["mean_velocity"])

            start = max(0.0, t + np.random.uniform(-rubato_amt, rubato_amt))
            duration = max(0.08, dur_seconds * legato)
            end = start + duration

            notes.append(
                pretty_midi.Note(
                    velocity=int(np.clip(velocity, 1, 127)),
                    pitch=int(np.clip(pitch, 0, 127)),
                    start=float(start),
                    end=float(end),
                )
            )

            if random.random() < 0.18:
                harm_pitch = pitch + random.choice([3, 4, 7])
                if register_low <= harm_pitch <= register_high:
                    harm_duration = max(0.05, duration * 0.97)
                    harm_end = start + harm_duration

                    notes.append(
                        pretty_midi.Note(
                            velocity=int(np.clip(velocity - 8, 1, 127)),
                            pitch=int(np.clip(harm_pitch, 0, 127)),
                            start=float(start),
                            end=float(harm_end),
                        )
                    )

            prev_pitch = pitch
            t += dur_seconds

            if t > bars * 4 * beat_seconds:
                break

        return notes

    def _sample_pitch(
        self,
        scale_pitches: list[int],
        pitch_weights: np.ndarray,
        prev_pitch: Optional[int],
        register_low: int,
        register_high: int,
        temperature: float,
    ) -> int:
        candidates = np.array(scale_pitches, dtype=np.int32)
        weights = pitch_weights.astype(np.float64).copy()

        if prev_pitch is not None:
            leap_penalty = np.array([abs(int(p) - int(prev_pitch)) for p in candidates], dtype=np.float64)
            weights *= np.exp(-leap_penalty / 7.5)

        # sharpen / flatten randomness
        temp = max(0.2, float(temperature))
        weights = np.power(weights, 1.0 / temp)
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum()

        pitch = int(np.random.choice(candidates, p=weights))
        pitch = max(register_low, min(register_high, pitch))
        return pitch

    def _pitch_weights_for_scale(self, scale_pitches: list[int], context_profile: dict) -> np.ndarray:
        hist = context_profile["pitch_hist"]
        center = context_profile["register_center"]

        weights = []
        for pitch in scale_pitches:
            pc_weight = hist[pitch % 12]
            reg_weight = math.exp(-abs(pitch - center) / 10.0)
            weights.append(pc_weight * reg_weight)

        weights = np.array(weights, dtype=np.float64)
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum()
        return weights

    def _sample_duration_beats(self, mood: str, style: str) -> float:
        mood = mood.lower()
        style = style.lower()

        if mood in {"melancholic", "gentle", "calm"}:
            choices = [0.5, 1.0, 1.5, 2.0]
            probs = [0.15, 0.40, 0.25, 0.20]
        elif mood in {"dramatic", "tense"}:
            choices = [0.25, 0.5, 1.0]
            probs = [0.35, 0.45, 0.20]
        else:
            choices = [0.25, 0.5, 1.0, 1.5]
            probs = [0.20, 0.40, 0.30, 0.10]

        if "baroque" in style:
            choices = [0.25, 0.5, 1.0]
            probs = [0.40, 0.45, 0.15]

        return float(np.random.choice(choices, p=probs))

    def _sample_velocity(self, mood: str, context_velocity: float) -> int:
        base = int(context_velocity)

        if mood == "melancholic":
            base -= 8
        elif mood == "gentle":
            base -= 12
        elif mood == "dramatic":
            base += 12
        elif mood == "tense":
            base += 8
        elif mood == "joyful":
            base += 6

        velocity = int(np.clip(np.random.normal(loc=base, scale=8), 30, 110))
        return velocity

    def _compose_accompaniment(
    self,
    bpm: float,
    bars: int,
    beat_seconds: float,
    root_name: str,
    mode: str,
    mood: str,
    style: str,
    notes_text: str,
    context_profile: dict,
    density: float,
    legato: float,
    rubato_amt: float,
    polyphony_prob: float,
) -> list[pretty_midi.Note]:
        notes = []
        root_pc = self.NOTE_NAMES.index(root_name)

        progression = self._make_progression(root_pc, mode, bars)
        bass_register = (36, 60)

        arpeggiated = (
            "arpeggio" in notes_text
            or "arpeggios" in notes_text
            or "nocturne" in notes_text
            or style in {"chopin", "debussy"}
        )

        for bar_idx in range(bars):
            chord_pcs = progression[bar_idx % len(progression)]
            bar_start = bar_idx * 4 * beat_seconds

            if arpeggiated:
                pattern = [0, 1, 2, 1, 0, 1, 2, 1]
                step = beat_seconds / 2.0

                for i, chord_idx in enumerate(pattern):
                    start = bar_start + i * step + np.random.uniform(-rubato_amt / 2, rubato_amt / 2)
                    start = max(0.0, start)

                    duration = max(0.05, step * max(0.65, legato))
                    end = start + duration

                    pitch = self._choose_chord_tone(chord_pcs, octave_base=48, which=chord_idx)
                    velocity = int(np.clip(np.random.normal(loc=context_profile["mean_velocity"] - 12, scale=6), 28, 92))

                    notes.append(
                        pretty_midi.Note(
                            velocity=int(np.clip(velocity, 1, 127)),
                            pitch=int(np.clip(pitch, 0, 127)),
                            start=float(start),
                            end=float(end),
                        )
                    )
            else:
                # block or bass-chord accompaniment
                root_pitch = 36 + chord_pcs[0]
                start = max(0.0, bar_start)
                duration = max(0.1, 4 * beat_seconds * max(0.8, legato))
                end = start + duration

                notes.append(
                    pretty_midi.Note(
                        velocity=int(np.clip(context_profile["mean_velocity"] - 10, 28, 95)),
                        pitch=int(np.clip(root_pitch, 0, 127)),
                        start=float(start),
                        end=float(end),
                    )
                )

                if random.random() < polyphony_prob:
                    for pc in chord_pcs[1:]:
                        chord_pitch = 48 + pc
                        chord_duration = max(0.05, duration * 0.95)
                        chord_end = start + chord_duration

                        notes.append(
                            pretty_midi.Note(
                                velocity=int(np.clip(context_profile["mean_velocity"] - 16, 24, 85)),
                                pitch=int(np.clip(chord_pitch, 0, 127)),
                                start=float(start + 0.02),
                                end=float(max(start + 0.07, chord_end)),
                            )
                        )

        return notes

    def _make_progression(self, root_pc: int, mode: str, bars: int) -> list[list[int]]:
        # simple diatonic triads in pitch classes
        if mode == "minor":
            i = [root_pc % 12, (root_pc + 3) % 12, (root_pc + 7) % 12]
            iv = [(root_pc + 5) % 12, (root_pc + 8) % 12, (root_pc + 0) % 12]
            v = [(root_pc + 7) % 12, (root_pc + 10) % 12, (root_pc + 2) % 12]
            vi = [(root_pc + 8) % 12, (root_pc + 0) % 12, (root_pc + 3) % 12]
            progression = [i, iv, i, v, i, vi, iv, v]
        else:
            I = [root_pc % 12, (root_pc + 4) % 12, (root_pc + 7) % 12]
            IV = [(root_pc + 5) % 12, (root_pc + 9) % 12, (root_pc + 0) % 12]
            V = [(root_pc + 7) % 12, (root_pc + 11) % 12, (root_pc + 2) % 12]
            vi = [(root_pc + 9) % 12, (root_pc + 0) % 12, (root_pc + 4) % 12]
            progression = [I, IV, I, V, I, vi, IV, V]

        return progression[:max(4, min(len(progression), bars))]

    def _choose_chord_tone(self, chord_pcs: list[int], octave_base: int, which: int = 0) -> int:
        pc = chord_pcs[which % len(chord_pcs)]
        return octave_base + pc

    def _build_scale_pitches(self, root_name: str, mode: str, low: int, high: int) -> list[int]:
        root_pc = self.NOTE_NAMES.index(root_name)
        intervals = self.MINOR_INTERVALS if mode == "minor" else self.MAJOR_INTERVALS

        pitches = []
        for midi_pitch in range(low, high + 1):
            if (midi_pitch - root_pc) % 12 in intervals:
                pitches.append(midi_pitch)

        if not pitches:
            pitches = list(range(low, high + 1))
        return pitches

    def _midi_to_bytes(self, midi: pretty_midi.PrettyMIDI) -> bytes:
        """
        pretty_midi writes to file paths, so use a temporary file and read bytes back.
        """
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            midi.write(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)