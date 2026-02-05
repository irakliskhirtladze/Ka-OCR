from functools import partial
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    default_data_collator)

from ml_training.dataset import GeorgianTokenizer, GeorgianOCRDataset
from ml_training.setup import setup_environment, check_env
from ml_training.training.training_loop import train_model, save_final_model
# from ml_training.training.validation import compute_metrics_seq2seq


def main() -> None:
    paths = setup_environment()

    # Set up dataframes
    df = pd.read_csv(paths.dataset_dir / "metadata.csv")
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # set up processor and tokenizer
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    tokenizer = GeorgianTokenizer(max_length=32)

    # Load model and resize token embeddings
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    for param in model.encoder.parameters():
        param.requires_grad = False  # Freeze entire encoder part of the model
    model.decoder.resize_token_embeddings(len(tokenizer))  # Resize to 37

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

    # from ml_training.utils import save_debug_samples
    # save_debug_samples(train_loader, tokenizer, str(paths.drive_output_dir / "debug_augs"))

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
        epochs=3,
        save_every=1000,
        max_grad_norm=1.0,
        learning_rate=1e-6,
        resume_latest=True
    )
    save_final_model(paths, model, processor, tokenizer)

    # # ============= seq2seqtrainer version =============
    # seq2seq_training_args = Seq2SeqTrainingArguments(
    #     predict_with_generate=True,
    #
    #     eval_strategy='steps',
    #     eval_steps=1000,
    #     logging_strategy="steps",
    #     logging_steps=1000,
    #
    #     save_strategy='steps',
    #     save_steps=1000,
    #     save_total_limit=2,
    #
    #     load_best_model_at_end=True,
    #     metric_for_best_model='cer',
    #     greater_is_better=False,
    #
    #     learning_rate=1e-6,
    #     weight_decay=0.01,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     fp16=True,
    #     output_dir=str(paths.output_dir),
    #     num_train_epochs=3,
    #     report_to='tensorboard',
    # )
    #
    # metrics_with_tokenizer = partial(compute_metrics_seq2seq, tokenizer=tokenizer)
    # seq2sec_trainer = Seq2SeqTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     args=seq2seq_training_args,
    #     compute_metrics=metrics_with_tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     data_collator=default_data_collator
    # )
    #
    # print("Starting Seq2Seq Training...")
    # result = seq2sec_trainer.train()
    # seq2sec_trainer.save_model(str(paths.output_dir / "best_model_final"))


if __name__ == "__main__":
    main()
