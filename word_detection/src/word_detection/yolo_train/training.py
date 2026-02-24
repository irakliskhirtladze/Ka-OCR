from ultralytics import YOLO

from word_detection.utils import PATHS


def train(resume: bool = False) -> None:
    # Always look locally
    model_path = PATHS.checkpoints_dir / "last.pt" if resume else "yolo26n.pt"
    model = YOLO(str(model_path))

    model.train(
        data=PATHS.base_dir / "ka_dataset.yaml",
        epochs=100,
        imgsz=640,
        device=0,  # GPU index
        project=str(PATHS.output_dir),
        name="ka_word_detector",
        resume=resume
    )
