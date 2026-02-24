import torch
from ultralytics import YOLO

from word_detection.utils import PATHS, sync_to_drive, check_env


def on_train_epoch_end(trainer) -> None:
    """Callback triggered at the end of every training epoch."""
    print(f"\nEpoch {trainer.epoch + 1} finished. Syncing to Drive...")
    if check_env() == "colab":
        sync_to_drive()


def train(resume: bool = False) -> None:
    device_id = 0 if torch.cuda.is_available() else "cpu"

    # Always look locally
    model_path = PATHS.checkpoints_dir / "last.pt" if resume else "yolo26s.pt"
    model = YOLO(str(model_path))

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data=PATHS.base_dir / "ka_dataset.yaml",
        epochs=100,
        imgsz=1024,  # Increased from 640 to better handle larger images
        scale=0.5,  # Multi-scale training: ±50% image size variation
        device=device_id,  # GPU index
        project=str(PATHS.output_dir),
        name="ka_word_detector",
        resume=resume,
        save=True,
        box=10,  # Increased from default 7.5 to penalize missed detections more
        cls=0.5,  # Keep classification loss lower
        dfl=1.5,   # Keep distribution focal loss moderate
        batch=-1
    )
