import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import TrOCRProcessor, PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split

from ml_training.dataset import GeorgianOCRDataset
from ml_training.setup import Paths, check_env
from ml_training.training.training_loop import train_model


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

    # State variables for the trainer
    resume_from_checkpoint = True

    if stage == 1:
        print("\n--- STAGE 1: PRETRAINING ---")
        train_df, val_df = train_test_split(synth_df, test_size=0.01, random_state=42)

        lr = 1e-5
        epochs = 10
        batch_size = 16
        sampler = None  # Standard shuffle is fine for synth-only
        checkpoint_prefix = "s1"

    else:
        print("\n--- STAGE 2: FINETUNING ---")
        real_df = pd.read_csv(paths.dataset_dir / "real" / "metadata.csv")
        real_df['file_name'] = "real/" + real_df['file_name'].astype(str)

        # Split real data
        real_train, real_val = train_test_split(real_df, test_size=0.5, random_state=42)

        # Grab the refresher
        synth_refresher = synth_df.sample(2000, random_state=42)

        # Combine
        train_df = pd.concat([real_train, synth_refresher], ignore_index=True)
        val_df = real_val  # Validate only on real data in Stage 2

        lr = 5e-6  # smaller for finetuning
        epochs = 30
        batch_size = 8
        checkpoint_prefix = "s2"

        # Setup Weighted Sampler to ensure Real images are seen often
        weights = [10.0] * len(real_train) + [1.0] * len(synth_refresher)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)

        # AUTOMATIC WEIGHT LOADING LOGIC
        # Check if we already have a Stage 2 checkpoint (if so, we just resume)
        s2_checkpoints = list(paths.output_dir.glob("checkpoint_s2_*.pt"))
        if not s2_checkpoints:
            # We are starting Stage 2 for the first time. Load Stage 1 Best Model.
            best_s1 = paths.drive_output_dir / "best_model.pt"
            if best_s1.exists():
                print(f"First run of Stage 2. Loading Stage 1 peaks weights from {best_s1}")
                model.load_state_dict(torch.load(best_s1, map_location='cpu'))
                resume_from_checkpoint = False  # Don't try to load optimizer state from Stage 1
            else:
                print("Warning: Stage 2 started but no best_model.pt found from Stage 1!")

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
        stage_prefix=checkpoint_prefix,
        epochs=epochs,
        learning_rate=lr,
        save_every=1000,
        resume_latest=resume_from_checkpoint,
    )
