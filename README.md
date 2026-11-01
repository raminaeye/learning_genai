# Learning GenAI: Lightweight Fine-Tuning

Fine-tunes DistilBERT to label Yelp reviews as good or bad. It first trains only the classification head, then adds LoRA adapters and trains further.

## What's inside

- `genai_fundamental/llm_peft_hgf/LightweightFineTuning.ipynb` is the notebook

## How to run

```bash
pip install -r requirements.txt "transformers==4.50.3"
jupyter notebook genai_fundamental/llm_peft_hgf/LightweightFineTuning.ipynb
```

## Related tutorial

- [Adapters and LoRA](https://raminaeye.github.io/adapter.html)
