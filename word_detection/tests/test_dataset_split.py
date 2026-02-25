import itertools

import pytest

from word_detection.utils import PATHS


def _all_font_files() -> list:
    return sorted(
        itertools.chain(PATHS.fonts_dir.glob("*.ttf"), PATHS.fonts_dir.glob("*.otf"))
    )


def test_train_val_dirs_exist():
    assert PATHS.train_dir.exists()
    assert PATHS.val_dir.exists()


def test_split_ratio():
    font_files = _all_font_files()
    assert len(font_files) > 0, "No font files found"

    split_idx = int(len(font_files) * 0.9)
    train_count = split_idx
    val_count = len(font_files) - split_idx

    expected_val = len(font_files) * 0.1
    assert abs(val_count - expected_val) <= 1, (
        f"Val split off: got {val_count}, expected ~{expected_val:.1f}"
    )
    assert train_count > val_count


def test_no_font_overlap_between_splits():
    font_files = _all_font_files()
    split_idx = int(len(font_files) * 0.9)
    train_stems = {f.stem for f in font_files[:split_idx]}
    val_stems = {f.stem for f in font_files[split_idx:]}

    overlap = train_stems & val_stems
    assert not overlap, f"Font(s) appear in both splits: {overlap}"