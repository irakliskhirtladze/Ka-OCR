from word_detection.utils import PATHS, setup_environment
from word_detection.yolo_train.training_loop import train


def main() -> None:
    setup_environment()
    train()


if __name__ == '__main__':
    main()
