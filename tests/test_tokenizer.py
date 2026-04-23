import pytest
import pretty_midi
from src.tokenizer import MIDITokenizer, TokenizerConfig

@pytest.fixture
def tokenizer():
    """Provides a fresh tokenizer with default config for each test."""
    return MIDITokenizer()

def create_simple_midi(pitch=60, start=0.0, end=1.0, velocity=100):
    """Helper to create a 1-note MIDI object in memory."""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
    piano.notes.append(note)
    midi.instruments.append(piano)
    return midi

def test_bos_eos_tokens(tokenizer):
    """Check if BOS and EOS tokens are added correctly."""
    midi = create_simple_midi()
    tokens = tokenizer.tokenize_midi(midi)
    
    assert tokens[0] == tokenizer.BOS
    assert tokens[-1] == tokenizer.EOS

def test_pitch_encoding(tokenizer):
    """Verify that a specific pitch (Middle C = 60) encodes to the correct ID."""
    midi = create_simple_midi(pitch=60)
    tokens = tokenizer.tokenize_midi(midi)
    
    # In your logic: note_on_base + pitch
    # 4 (time_shift_base) + 100 (max_time_shift) + 60 = 164
    expected_pitch_token = tokenizer.note_on_base + 60
    assert expected_pitch_token in tokens

def test_time_shift_accumulation(tokenizer):
    """Test that a long silence is broken into multiple TimeShift tokens."""
    # 2.5 seconds gap / 0.01 step = 250 steps. 
    # Max shift is 100, so we expect two 100s and one 50.
    midi = create_simple_midi(start=2.5, end=3.5) 
    tokens = tokenizer.tokenize_midi(midi)
    
    # Check that we have multiple time shift tokens before the pitch token
    time_tokens = [t for t in tokens if tokenizer.time_shift_base <= t < tokenizer.note_on_base]
    assert len(time_tokens) == 3 # 100, 100, 50

def test_velocity_binning(tokenizer):
    """Verify that different velocities map to distinct bins."""
    low_vel_midi = create_simple_midi(velocity=10)
    high_vel_midi = create_simple_midi(velocity=120)
    
    low_tokens = tokenizer.tokenize_midi(low_vel_midi)
    high_tokens = tokenizer.tokenize_midi(high_vel_midi)
    
    # Get velocity tokens (they come after pitch)
    def get_vel(tks): return [t for t in tks if tokenizer.velocity_base <= t < tokenizer.duration_base][0]
    
    assert get_vel(low_tokens) != get_vel(high_tokens)
    assert get_vel(high_tokens) > get_vel(low_tokens)

def test_empty_midi(tokenizer):
    """An empty MIDI should only return BOS and EOS."""
    midi = pretty_midi.PrettyMIDI()
    tokens = tokenizer.tokenize_midi(midi)
    assert tokens == [tokenizer.BOS, tokenizer.EOS]