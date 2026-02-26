import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)

from ml_training.dataset import GeorgianOCRDataset
from ml_training.setup import setup_environment, check_env
from ml_training.training.training_loop import train_model
from ml_training.training.training_stages import run_training_stage
from ml_training.utils import save_debug_samples, save_final_model, upload_to_hf


def main() -> None:
    paths = setup_environment()

    # # set up processor and tokenizer
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    # tokenizer = processor.tokenizer
    # georgian_chars = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
    # tokenizer.add_tokens(georgian_chars)
    #
    # # Load model and resize token embeddings
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    # model.decoder.resize_token_embeddings(len(tokenizer))
    #
    # # Configure special tokens
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.eos_token_id = tokenizer.sep_token_id

    # run_training_stage(2, paths, model, processor, tokenizer)
    # save_final_model(paths, model, processor)
    upload_to_hf(paths)

    # # ============= seq2seqtrainer version =============
    # print("Starting Seq2Seq Training...")
    # from ml_training.training.seq2seq import init_seq2seq_trainer
    # seq2sec_trainer = init_seq2seq_trainer()
    # seq2sec_trainer.train()
    # seq2sec_trainer.save_model(str(paths.output_dir / "best_model_final"))


    # # ================ INFERENCE ====================
    # from ml_training.utils import test_against_real_images
    # test_against_real_images(paths, model, processor, tokenizer)


if __name__ == "__main__":
    main()
