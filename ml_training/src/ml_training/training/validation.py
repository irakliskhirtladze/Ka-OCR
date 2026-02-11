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
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    pad_token_id=model.config.pad_token_id,
                    eos_token_id=model.config.eos_token_id,
                )

                # Convert tokens back to strings
                pred_str = [tokenizer.decode(ids.tolist()) for ids in outputs]

                # Convert label tokens back to strings (ignoring -100 padding)
                labels_copy = labels.clone()  # Clone so we don't mess up the original data
                labels_copy[labels_copy == -100] = tokenizer.pad_token_id
                label_str = [tokenizer.decode(ids.tolist()) for ids in labels_copy]

                predictions.extend(pred_str)
                references.extend(label_str)

            print("eos_token_id:", tokenizer.eos_token_id)
            print("first output ids:", outputs[0].tolist())
            print(predictions[:5])
            print(references[:5])

    # Calculate Character Error Rate
    return cer_metric.compute(predictions=predictions, references=references)
