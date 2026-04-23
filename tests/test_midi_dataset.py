from pathlib import Path

import pytest
import torch

from src.midi_dataset import MIDIDataset


def test_init_raises_for_missing_raw_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        MIDIDataset(raw_dir=str(missing))


def test_collects_mid_and_midi_files(tmp_path: Path) -> None:
    year_2004 = tmp_path / "2004"
    year_2006 = tmp_path / "2006"
    year_2004.mkdir(parents=True)
    year_2006.mkdir(parents=True)

    (year_2004 / "a.mid").write_bytes(b"")
    (year_2004 / "ignore.txt").write_text("x")
    (year_2006 / "b.midi").write_bytes(b"")

    ds = MIDIDataset(raw_dir=str(tmp_path), max_seq_len=8)
    assert len(ds) == 2


def test_getitem_truncates_and_pads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    year_2004 = tmp_path / "2004"
    year_2004.mkdir(parents=True)
    (year_2004 / "sample.mid").write_bytes(b"")

    ds = MIDIDataset(raw_dir=str(tmp_path), max_seq_len=5)

    # Long sequence is truncated.
    monkeypatch.setattr(ds, "tokenize", lambda _path: [1, 2, 3, 4, 5, 6, 7])
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.long
    assert x.tolist() == [1, 2, 3, 4, 5]

    # Short sequence is padded with zeros.
    monkeypatch.setattr(ds, "tokenize", lambda _path: [9, 8])
    y = ds[0]
    assert y.tolist() == [9, 8, 0, 0, 0]
