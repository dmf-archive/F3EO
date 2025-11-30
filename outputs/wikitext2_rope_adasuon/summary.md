# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "type": "rope",
    "vocabulary_size": 40479,
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 6,
    "num_layers": 4,
    "rope_theta": 10000.0,
    "intermediate_size": 2048,
    "tie_word_embeddings": true
  },
  "data": {
    "batch_size": 8,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "AdaSuon",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2 | 6.5858 | 0.000100 | 0.366 | N/A | 0.000 | 1.0043 | 600.33 | 2943.8 | 6.12 | 454.31 |
| 2 | wikitext2 | 5.7144 | 0.000100 | 0.410 | N/A | 0.000 | 0.8924 | 599.37 | 2943.8 | 5.76 | 317.71 |
| 3 | wikitext2 | 5.3688 | 0.000100 | 0.387 | N/A | 0.000 | 0.9497 | 595.54 | 2943.8 | 5.69 | 296.63 |
| 4 | wikitext2 | 6.3339 | 0.000100 | 0.088 | N/A | 0.000 | 2.4280 | 591.22 | 2943.8 | 7.83 | 2513.23 |
| 5 | wikitext2 | 9.1411 | 0.000100 | 0.006 | N/A | 0.000 | 5.1071 | 612.48 | 2943.8 | 9.70 | 16376.52 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 9.70, wikitext2 Perplexity: 296.63
- **Final Validation Metrics**: wikitext2: {"loss": 9.703604056917388, "perplexity": 16376.52288708862}
