import gc
import shutil

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from ml_training.dataset import GeorgianTokenizer
from ml_training.setup import Paths, check_env
from checkpoints import load_latest_state, save_state


def train_model(
        paths: Paths,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loader_generator: torch.Generator,
        device: torch.device,
        tokenizer: GeorgianTokenizer,
        epochs: int = 3,
        save_every: int = 1000,
        max_grad_norm: float = 1.0,
        learning_rate: float = 1e-5,
        resume_latest: bool = True
) -> None:
    """Main pytorch training loop to fine-tune TrOCR pretrained model."""

    # Always create fresh optimizer and scaler to avoid OOM from stale state
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    # Initialize defaults
    start_epoch = 0
    start_batch = -1
    best_cer = float('inf')

    if resume_latest:
        # Load model weights and RNG states only (not optimizer/scaler)
        start_epoch, start_batch, checkpoint_data = load_latest_state(
            model, optimizer, scaler, loader_generator,
            load_optimizer=False, load_scaler=False
        )

        if checkpoint_data and 'cer' in checkpoint_data:
            best_cer = checkpoint_data['cer']
            print(f"Resumed Best CER: {best_cer:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        print(f"\n--- Starting Epoch {epoch} ---")

        train_iter = iter(train_loader)
        batch_offset = 0

        # Fast-forward past already-trained batches (only on resume epoch)
        if epoch == start_epoch and start_batch >= 0:
            skip_count = start_batch + 1
            print(f"Fast-forwarding past {skip_count} batches...")
            for _ in range(skip_count):
                try:
                    next(train_iter)
                except StopIteration:
                    break
            batch_offset = skip_count

        # Continue training from where we left off
        for batch_idx, batch in enumerate(train_iter, start=batch_offset):
            try:
                pixel_values: torch.Tensor = batch["pixel_values"].to(device)
                labels: torch.Tensor = batch["labels"].to(device)

                with autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss: torch.Tensor = outputs.loss

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

                if batch_idx > 0 and batch_idx % save_every == 0:
                    checkpoint_name = f"checkpoint_e{epoch}_b{batch_idx}.pt"
                    save_state(epoch, batch_idx, model, optimizer, scaler, loss.item(), best_cer, checkpoint_name,
                               loader_generator)
                    print(f"===== Saved Resume Checkpoint {checkpoint_name} =====")

                    print(f"Validating at Batch {batch_idx}...")
                    current_cer = validate_model(model, test_loader, tokenizer, device)
                    print(f"Batch {batch_idx} CER: {current_cer:.4f}")

                    if current_cer < best_cer:
                        best_cer = current_cer
                        print(f"New Best Model found! (CER: {current_cer:.4f}) Saving best_model.pt...")

                        torch.save(model.state_dict(), paths.output_dir / "best_model.pt")
                        if check_env() == "colab":
                            shutil.copy(paths.output_dir / "best_model.pt", paths.drive_output_dir / "best_model.pt")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU OOM detected at batch {batch_idx}. Cleaning memory and skipping...")

                    if 'outputs' in locals():
                        del outputs
                    if 'loss' in locals():
                        del loss
                    del pixel_values, labels

                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # --- END OF EPOCH SAVE
        print(f"Epoch {epoch} complete. Performing final validation and save...")
        current_cer = validate_model(model, test_loader, tokenizer, device)

        save_state(epoch, batch_idx, model, optimizer, scaler, loss.item(),
                   current_cer, f"checkpoint_e{epoch}_final.pt", loader_generator)

        if current_cer < best_cer:
            best_cer = current_cer
            torch.save(model.state_dict(), paths.output_dir / "best_model.pt")
            if check_env() == "colab":
                shutil.copy(paths.output_dir / "best_model.pt", paths.drive_output_dir / "best_model.pt")
