# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2_line"
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
    "name": "ARS",
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "rho": 0.1
  },
  "train": {
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2_line | 5.1701 | 0.001000 | 0.525 | N/A | 0.000 | 0.6434 | 783.60 | 2949.4 | 4.83 | 124.60 |
| 2 | wikitext2_line | 4.2192 | 0.001000 | 0.587 | N/A | 0.000 | 0.5322 | 787.99 | 3261.5 | 4.57 | 96.71 |
| 3 | wikitext2_line | 3.7296 | 0.001000 | 0.584 | N/A | 0.000 | 0.5384 | 785.59 | 3254.6 | 4.52 | 92.14 |
| 4 | wikitext2_line | 3.3322 | 0.001000 | 0.566 | N/A | 0.000 | 0.5683 | 787.72 | 3264.6 | 4.59 | 98.95 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 4.83, wikitext2_line Perplexity: 92.14
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.594612988073434, "perplexity": 98.94983337080015}
