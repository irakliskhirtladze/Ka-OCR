from word_detection.utils import PATHS, setup_environment, create_yaml
from word_detection.yolo_train.handle_model import model_to_hf, save_tested_predictions
from word_detection.yolo_train.training import train


def main() -> None:
    # create_yaml()
    # setup_environment()
    # train()
    # model_to_hf()
    save_tested_predictions(conf_labels=False)


if __name__ == '__main__':
    main()
