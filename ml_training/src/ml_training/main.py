import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)

from ml_training.dataset import GeorgianTokenizer, GeorgianOCRDataset
from ml_training.setup import setup_environment, check_env
from ml_training.training.training_loop import train_model, save_final_model


def main() -> None:
    paths = setup_environment()

    # Set up dataframes
    df = pd.read_csv(paths.dataset_dir / "metadata.csv")
    train_df, test_df = train_test_split(
        df,
        test_size=0.05,
        random_state=42,
        shuffle=True
    )

    # set up processor and tokenizer
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    tokenizer = GeorgianTokenizer(max_length=32)

    # Load model and resize token embeddings
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    # for param in model.encoder.parameters():
    #     param.requires_grad = False  # Freeze entire encoder part of the model
    model.decoder.resize_token_embeddings(len(tokenizer))

    # Configure special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # set up datasets and loader generators
    train_dataset = GeorgianOCRDataset(train_df, str(paths.dataset_dir), processor, tokenizer, augment=True)
    test_dataset = GeorgianOCRDataset(test_df, str(paths.dataset_dir), processor, tokenizer)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=loader_generator,
                              num_workers=2 if check_env() == "colab" else 6)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # # Quick check of online augmented samples
    # print("saving online augmented images for quick check...")
    # from ml_training.utils import save_debug_samples
    # if check_env() == "colab":
    #     save_debug_samples(train_loader, tokenizer, str(paths.drive_output_dir / "debug_augs"), n_images=1000)
    # elif check_env() == "local":
    #     save_debug_samples(train_loader, tokenizer, str(paths.output_dir / "debug_augs"), n_images=1000)

    # set up device and run training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_model(
        paths,
        model,
        train_loader,
        test_loader,
        loader_generator,
        device,
        tokenizer,
        epochs=20,
        save_every=1000,
        max_grad_norm=1.0,
        learning_rate=1e-5,
        resume_latest=True
    )
    save_final_model(paths, model, processor, tokenizer)


    # # ============= seq2seqtrainer version =============
    # print("Starting Seq2Seq Training...")
    # from ml_training.training.seq2seq import init_seq2seq_trainer
    # seq2sec_trainer = init_seq2seq_trainer()
    # seq2sec_trainer.train()
    # seq2sec_trainer.save_model(str(paths.output_dir / "best_model_final"))


    # ================ INFERENCE ====================
    # from ml_training.utils import test_against_real_images
    # test_against_real_images(paths, model, processor, tokenizer)


if __name__ == "__main__":
    main()
