# Ka-OCR
This is a monorepo of OCR project that is able to detect Georgian (Ka)
texts on images/PDFs.

The project consists of few parts/subprojects:
## dataset_gen
Handles synthetic data generation, adding real image data and it's augmentation,
unified metadate.csv generation, zipping and uploading
to Hugging Face automatically.

## ml_training
Runs model training script and manages checkpoints, model evaluation and best model saving.

## api
FastAPI based user-facing api for model serving.


# Setup
Each subproject is independently managed with UV.
Go to the subproject root you want to run/edit and run command to 
automatically setup venv and install deps.

For example, in case of ml_training subproject:
```bash
cd ml_training
uv sync
```
Then run respective entry point:
```bash
uv run main
```
See more detailed information about setup and usage in each subproject's readme file.
