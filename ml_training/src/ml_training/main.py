from ml_training.setup import setup_environment


def main() -> None:
    paths = setup_environment()
    for item in paths.dataset_dir.iterdir():
        print(item)


if __name__ == "__main__":
    main()
