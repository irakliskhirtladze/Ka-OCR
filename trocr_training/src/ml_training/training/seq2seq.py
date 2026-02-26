import torch.nn
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          default_data_collator, EvalPrediction)
from functools import partial
import evaluate

from ml_training.dataset import GeorgianTokenizer, GeorgianOCRDataset
from ml_training.setup import Paths

cer_metric = evaluate.load("cer")


def init_args(paths: Paths) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        predict_with_generate=True,

        eval_strategy='steps',
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,

        save_strategy='steps',
        save_steps=1000,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,

        learning_rate=1e-6,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        output_dir=str(paths.output_dir),
        num_train_epochs=3,
        report_to='tensorboard',
    )


def compute_metrics_seq2seq(pred: EvalPrediction, tokenizer: GeorgianTokenizer) -> dict[str, float]:
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions and labels
    # -100 is the ignore index we set in our Dataset
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(labels_ids)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def init_seq2seq_trainer(tokenizer: GeorgianTokenizer, model: torch.nn.Module, seq2seq_args: Seq2SeqTrainingArguments,
                         train_dataset: GeorgianOCRDataset, test_dataset: GeorgianOCRDataset) -> Seq2SeqTrainer:
    metrics_with_tokenizer = partial(compute_metrics_seq2seq, tokenizer=tokenizer)
    return Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=seq2seq_args,
        compute_metrics=metrics_with_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator
    )
