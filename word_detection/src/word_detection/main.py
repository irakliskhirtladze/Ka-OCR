from word_detection.utils import PATHS, setup_environment, create_yaml
from word_detection.yolo_train.training import train


def main() -> None:
    create_yaml()
    setup_environment()
    train()


if __name__ == '__main__':
    main()
