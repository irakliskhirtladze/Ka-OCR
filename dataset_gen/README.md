# Georgian Text-on-Image Synthetic Dataset Generator

Generate synthetic Georgian text images for OCR training.

The program uses dictionary of >100000 unique Georgian words,
which were extracted from scientific and fictional texts and from wikipedia
pages.

Words have weights based on their frequency in abovementioned sources.
The weights are used during image generation to reflect natural 
frequency in the language.

Dictionary files are located at: src/generator/dictionaries/

Additionally, during generation ~7% of images will have random character
sequences to reflect typos in words and very rare or specific words.
~3% will have random numerical text.

## Setup
Using Python 3.10 and UV is strongly recommended to avoid dependency hell.

1. Open dataset_gen as standalone project in your IDE (assuming you use IDE).

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Create `.env` file** in project root:
   ```
   HF_TOKEN=your_huggingface_token
   HF_DATASET_REPO=your_username/your_dataset_name
   ```

   Get your HF token from: https://huggingface.co/settings/tokens

## Usage

### Run

```bash
python main.py
```

### What Happens

1. **Generation**: Creates synthetic images in `data/raw/` with metadata in `data/metadata.csv`
   - Uses 67 Georgian fonts
   - 90% real Georgian words (from 100k+ word dictionary)
   - 7% random character sequences
   - 3% numbers/dates (except 4 fonts that do not support them)
   - Program supports both sequential and parallel data generation.
   The latter can produce dataset few times faster, so it's generally recommended.

2. **Upload to HF** (optional): 
   - Prompts you to zip and upload to Hugging Face
   - Creates `data/ka-ocr.zip` with all images and metadata
   - Uploads to your HF dataset repo
   - Overwrites existing zip file if present

## Dataset Structure

```
data/
├── raw/                        # Generated images (gitignored)
│   ├── font1/
│   │   ├── font1_0000.png
│   │   ├── font1_0001.png
│   │   └── ...
│   ├── font2/
│   │   └── ...
│   └── ...
├── metadata.csv                # Image labels (gitignored)
└── ka-ocr.zip                  # Zipped dataset for HF (gitignored)
```
