from pathlib import Path

import cv2
from huggingface_hub import HfApi
from ultralytics import YOLO

from word_detection.utils import PATHS


def model_to_hf() -> None:
    """Uploads best.pt to Hugging Face Hub as yolo26s-ka-words.pt."""
    best_pt = PATHS.weights_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found at {best_pt}")

    upload_name = "yolo26s-ka-words.pt"
    repo_id = "irakliskhirtladze/YOLO-ka-word-detector"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(best_pt),
        path_in_repo=upload_name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded as {upload_name} to https://huggingface.co/{repo_id}")


def inference(conf: float = 0.25) -> list:
    """Run best.pt on all images in real_docs_dir. Returns list of ultralytics Results objects."""
    image_paths = [
        f for f in PATHS.real_docs_dir.glob("*")
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not image_paths:
        print(f"No images found in {PATHS.real_docs_dir}")
        return []

    model = YOLO(str(PATHS.weights_dir / "best.pt"))
    results = model.predict(image_paths, conf=conf, verbose=False)
    print(f"Ran inference on {len(image_paths)} images.")
    return results


def save_tested_predictions(conf_labels: bool = False) -> None:
    """Draw predicted bounding boxes on each image and save to real_docs/predictions/."""
    predictions_dir = PATHS.real_docs_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    results = inference()
    for result in results:
        img = result.orig_img.copy()  # BGR numpy array — opencv-ready

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if conf_labels:
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        out_path = predictions_dir / Path(result.path).name
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path.name} — {len(result.boxes)} words detected")
