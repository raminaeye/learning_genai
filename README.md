# Learning GenAI: Lightweight Fine-Tuning

Hands-on notebooks for learning generative AI fundamentals. The current notebook walks through lightweight (parameter-efficient) fine-tuning of a Hugging Face transformer for sentiment classification.

## Repo structure

- `genai_fundamental/llm_peft_hgf/LightweightFineTuning.ipynb` - fine-tune `distilbert-base-uncased` on Yelp reviews in two stages: first a classifier head on the frozen base model, then LoRA parameter-efficient fine-tuning, with accuracy comparison and error inspection.

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Open `genai_fundamental/llm_peft_hgf/LightweightFineTuning.ipynb` and run the cells in order.

Training uses a GPU when one is available and falls back to CPU otherwise (mixed precision is enabled only on CUDA). Fine-tuning downloads the model and dataset from Hugging Face, so an internet connection is needed on first run.

## What you learn

- Loading a pretrained Hugging Face model and tokenizer, and tokenizing a dataset.
- Training a classifier head on a frozen foundation model with the `Trainer` API.
- Applying LoRA (low-rank adaptation) to fine-tune with a tiny fraction of trainable parameters.
- Evaluating with custom metrics and inspecting misclassified examples.
- Saving and reloading both full and PEFT adapter weights.
