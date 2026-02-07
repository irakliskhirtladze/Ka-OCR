import cv2
import pandas as pd

from dataset_gen.utils import BASE_DIR


if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "data/real_augmented.csv")
    for file_path in (BASE_DIR / "data" / "real").glob("*.png"):
        cv_img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if cv_img.shape[0] < 30 or cv_img.shape[1] < 30:
            print(file_path)

