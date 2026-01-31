import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from ml_training.setup import Paths, check_env


def save_state(
        paths: Paths,
        epoch: int,
        batch_idx: int,
        model: torch.nn.Module,
        optimizer,
        scaler,
        loss,
        cer,
        filename,
        generator,
        keep=5):
    """
    Saves a resume checkpoint and maintains a rolling window of the last 'keep' files.
    Does NOT touch 'best_model.pt'.
    """
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'generator_state': generator.get_state(),
        'loss': loss,
        "cer": cer,
        'rng_state_torch': torch.get_rng_state(),
        'rng_state_numpy': np.random.get_state(),
        'rng_state_python': random.getstate(),
    }

    # Save Locally
    local_path = paths.checkpoint_dir / filename
    torch.save(checkpoint, local_path)

    if check_env() == "colab":  # Sync to Drive (Colab safe-write)
        drive_path = paths.drive_checkpoint_dir / filename
        # Write to temp file first, then move (atomic write) to prevent corruption
        temp_path = drive_path.with_suffix(".tmp")
        shutil.copy(local_path, temp_path)
        os.replace(temp_path, drive_path)
        print(f"Saved and synced to drive: {filename}")
        ckpts = sorted(paths.drive_checkpoint_dir.glob("checkpoint_e*_b*.pt"), key=os.path.getmtime)
    else:
        print(f"Saved locally: {filename}")
        ckpts = sorted(paths.checkpoint_dir.glob("checkpoint_e*_b*.pt"), key=os.path.getmtime)

    # Keep only last 'keep' number of checkpoints
    # We filter for files matching our pattern to avoid deleting 'best_model.pt'
    if len(ckpts) > keep:
        for old_ckpt in ckpts[:-keep]:
            old_ckpt.unlink()
            print(f"Deleted old checkpoint: {old_ckpt.name}")


def load_latest_state(paths, model, optimizer, scaler, generator, load_optimizer=False, load_scaler=False):
    """Load checkpoint. By default only loads model weights and RNG states (safe for resume)."""
    search_path = paths.drive_checkpoint_dir if check_env() == "colab" else paths.checkpoint_dir
    checkpoints: list[Path] = list(search_path.glob("checkpoint_e*_b*.pt"))

    if not checkpoints:
        print("No resume checkpoints found. Starting fresh.")
        return 0, -1, None

    latest_ckpt_path = max(checkpoints, key=os.path.getmtime)
    print(f"Resuming from: {latest_ckpt_path.name}")

    try:
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if load_scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        # RNG state restoration
        if 'rng_state_torch' in ckpt:
            rng_state = ckpt['rng_state_torch']
            if not isinstance(rng_state, torch.ByteTensor):
                rng_state = rng_state.to(torch.uint8)
            torch.set_rng_state(rng_state)

        if 'rng_state_numpy' in ckpt: np.random.set_state(ckpt['rng_state_numpy'])
        if 'rng_state_python' in ckpt: random.setstate(ckpt['rng_state_python'])
        if 'generator_state' in ckpt: generator.set_state(ckpt['generator_state'])

        return ckpt['epoch'], ckpt['batch_idx'], ckpt

    except Exception as e:
        print(f"Error loading checkpoint {latest_ckpt_path.name}: {e}")
        return 0, -1, None
