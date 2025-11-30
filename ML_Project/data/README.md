# Data Acquisition — Save Popular NLP Datasets to CSV

This minimal script downloads a dataset from the Hugging Face hub and saves splits to **CSV** with unified columns:
- `text`
- `label` (numeric)
- optional `label_text` (human‑readable)

Supported keys:
- `sst2` (GLUE/SST-2)
- `ag_news`
- `imdb`
- `tweeteval-sentiment`

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
**Save all SST-2 splits to ./data/**
```bash
python download_datasets.py --dataset sst2 --split all --outdir data --with_label_text
```

**Save AG News train split (first 50k examples)**
```bash
python download_datasets.py --dataset ag_news --split train --num_samples 50000 --outdir data
```

**Save IMDB test split**
```bash
python download_datasets.py --dataset imdb --split test --outdir data --with_label_text
```

**Save TweetEval (sentiment) all splits**
```bash
python download_datasets.py --dataset tweeteval-sentiment --split all --outdir data --with_label_text
```

## Notes
- The script automatically detects available splits; `--split all` saves everything present.
- Use `--num_samples` for quick smoke tests or few‑shot subsets.
