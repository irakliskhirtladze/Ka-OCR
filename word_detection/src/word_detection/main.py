import itertools

import cv2
from PIL import Image
from random import randint, choice

from word_detection.data_factory.augmentation import augment_doc, test_online_aug
from word_detection.data_factory.doc_gen import generate_docs, zip_dataset, dataset_to_hf
from word_detection.utils import PATHS
from word_detection.yolo_train.training_loop import train


def main() -> None:
    train()


if __name__ == '__main__':
    main()
