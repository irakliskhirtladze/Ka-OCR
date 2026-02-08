import evaluate
import torch
from transformers import EvalPrediction

from ml_training.dataset import GeorgianTokenizer


cer_metric = evaluate.load("cer")


def validate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: GeorgianTokenizer,
    device: torch.device,
    max_batches: int | None = 50,
    num_beams: int = 1,
) -> float:

    model.eval()
    predictions: list[str] = []
    references: list[str] = []

    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            for i, batch in enumerate(val_loader):
                if max_batches and i >= max_batches:  # Stop after enough samples
                    break

                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # Generate text from image
                outputs = model.generate(
                    pixel_values,
                    num_beams=num_beams,
                    max_length=32,
                )

                # Convert tokens back to strings
                pred_str = [tokenizer.decode(ids.tolist()) for ids in outputs]

                # Convert label tokens back to strings (ignoring -100 padding)
                labels_copy = labels.clone()  # Clone so we don't mess up the original data
                labels_copy[labels_copy == -100] = tokenizer.pad_token_id
                label_str = [tokenizer.decode(ids.tolist()) for ids in labels_copy]

                predictions.extend(pred_str)
                references.extend(label_str)

    # Calculate Character Error Rate
    return cer_metric.compute(predictions=predictions, references=references)


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
