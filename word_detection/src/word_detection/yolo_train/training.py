from ultralytics import YOLO

from word_detection.utils import PATHS, sync_to_drive, check_env


def on_train_epoch_end(trainer) -> None:
    """Callback triggered at the end of every training epoch."""
    print(f"\nEpoch {trainer.epoch + 1} finished. Syncing to Drive...")
    if check_env() == "colab":
        sync_to_drive()


def train(resume: bool = False) -> None:
    # Always look locally
    model_path = PATHS.checkpoints_dir / "last.pt" if resume else "yolo26n.pt"
    model = YOLO(str(model_path))

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data=PATHS.base_dir / "ka_dataset.yaml",
        epochs=100,
        imgsz=640,
        device=0,  # GPU index
        project=str(PATHS.output_dir),
        name="ka_word_detector",
        resume=resume,
        save=True
    )
