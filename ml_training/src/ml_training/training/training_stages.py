import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import TrOCRProcessor, PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split

from ml_training.dataset import GeorgianOCRDataset
from ml_training.setup import Paths, check_env
from ml_training.training.training_loop import train_model, save_final_model


def run_training_stage(stage: int,
                       paths: Paths,
                       model: torch.nn.Module,
                       processor: TrOCRProcessor,
                       tokenizer: PreTrainedTokenizerBase,
                       ) -> None:
    # Load and prepare dataframes. We load them early to fix the path issue immediately
    synth_df = pd.read_csv(paths.dataset_dir / "synthetic" / "metadata.csv")
    # Prefix the filenames so they point correctly from the parent dir
    synth_df['file_name'] = "synthetic/" + synth_df['file_name'].astype(str)

    if stage == 1:
        print("--- STAGE 1: PRETRAINING ---")
        train_df, val_df = train_test_split(synth_df, test_size=0.01, random_state=42)

        lr = 1e-5
        epochs = 10
        batch_size = 16
        sampler = None  # Standard shuffle is fine for synth-only

    else:
        print("--- STAGE 2: FINETUNING ---")
        real_df = pd.read_csv(paths.dataset_dir / "real" / "metadata.csv")
        real_df['file_name'] = "real/" + real_df['file_name'].astype(str)

        # Split real data
        real_train, real_val = train_test_split(real_df, test_size=0.5, random_state=42)

        # Grab the refresher
        synth_refresher = synth_df.sample(2000, random_state=42)

        # Combine
        train_df = pd.concat([real_train, synth_refresher], ignore_index=True)
        val_df = real_val  # Validate only on real data in Stage 2

        lr = 1e-6  # 10x smaller for finetuning
        epochs = 20
        batch_size = 8

        # Setup Weighted Sampler to ensure Real images are seen often
        weights = [10.0] * len(real_train) + [1.0] * len(synth_refresher)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)

    # Initialize Datasets. Note: root_dir is paths.dataset_dir
    train_dataset = GeorgianOCRDataset(
        train_df,
        str(paths.dataset_dir),
        processor,
        tokenizer,
        augment=True
    )

    validation_dataset = GeorgianOCRDataset(
        val_df,
        str(paths.dataset_dir),
        processor,
        tokenizer,
        augment=False
    )

    # Initialize Loaders. Note: If sampler is used, 'shuffle' must be False
    loader_generator = torch.Generator()
    loader_generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        generator=loader_generator,
        sampler=sampler,
        num_workers=2 if check_env() == "colab" else 6
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=loader_generator,
    )

    # Run Training
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
        epochs=epochs,
        learning_rate=lr,
        save_every=1000,
        resume_latest=True
    )

    save_final_model(paths, model, processor)
