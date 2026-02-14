import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)

from ml_training.dataset import GeorgianOCRDataset
from ml_training.setup import setup_environment, check_env
from ml_training.training.training_loop import train_model, save_final_model
from ml_training.utils import save_debug_samples


def main() -> None:
    paths = setup_environment()

    # Set up dataframes
    train_df = pd.read_csv(paths.dataset_dir / "synthetic" / "metadata.csv")

    real_data_df = pd.read_csv(paths.dataset_dir / "real" / "metadata.csv")
    validation_df, test_df = train_test_split(
        real_data_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    # set up processor and tokenizer
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    tokenizer = processor.tokenizer
    georgian_chars = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
    tokenizer.add_tokens(georgian_chars)

    # Load model and resize token embeddings
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    # for param in model.encoder.parameters():
    #     param.requires_grad = False  # Freeze entire encoder part of the model
    model.decoder.resize_token_embeddings(len(tokenizer))

    # Configure special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id

    # set up datasets and loader generators
    train_dataset = GeorgianOCRDataset(train_df, str(paths.dataset_dir / "synthetic"), processor, tokenizer,
                                       augment=True)
    validation_dataset = GeorgianOCRDataset(validation_df, str(paths.dataset_dir / "real"), processor, tokenizer,
                                            augment=True)
    test_dataset = GeorgianOCRDataset(test_df, str(paths.dataset_dir / "real"), processor, tokenizer)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=loader_generator,
                              num_workers=2 if check_env() == "colab" else 6)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # # quick check of augmented samples
    # save_debug_samples(train_loader, tokenizer, str(paths.output_dir), 100)

    # set up device and run training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_model(
        paths,
        model,
        train_loader,
        validation_loader,
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
