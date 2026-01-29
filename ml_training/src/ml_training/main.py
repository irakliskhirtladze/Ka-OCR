from ml_training.setup import prepare_dataset


def main() -> None:
    dataset_dir = prepare_dataset()
    for item in dataset_dir.iterdir():
        print(item)


if __name__ == "__main__":
    main()
