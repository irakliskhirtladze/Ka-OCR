from dataset_gen.generator.gen import generate_imgs, dataset_to_hf, zip_dataset


def main() -> None:
    # Image generation
    while True:
        user_input = input("\nHow many images per font would you like to generate?: ")
        if not user_input.isdigit() or int(user_input) <= 0:
            print("Please enter a positive integer.")
            continue
        else:
            break
    generate_imgs(int(user_input))

    # Zipping the dataset
    while True:
        user_input = input("\nDo you want to zip the dataset? (Y/N): ")
        if user_input.lower() == "y":
            zip_dataset()
            break
        elif user_input.lower() == "n":
            print("Zipping cancelled.")
            break
        else:
            print("Please enter either 'Y' or 'N'.")

    # Uploading dataset to HF
    while True:
        user_input = input("\nDo you want to upload zipped dataset ot Hugging Face? (Y/N) ?: ")
        if user_input.lower() == "y":
            dataset_to_hf()
            break
        elif user_input.lower() == "n":
            print("HF upload cancelled.")
            break
        else:
            print("Please enter either 'Y' or 'N'.")


if __name__ == "__main__":
    main()
