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
    "lr": 0.0001,
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
| 1 | wikitext2_line | 5.5777 | 0.000100 | 0.451 | N/A | 0.000 | 0.7952 | 784.80 | 2949.4 | 5.07 | 159.13 |
| 2 | wikitext2_line | 4.6183 | 0.000100 | 0.513 | N/A | 0.000 | 0.6675 | 795.82 | 3261.5 | 4.70 | 109.88 |
| 3 | wikitext2_line | 4.1922 | 0.000100 | 0.498 | N/A | 0.000 | 0.6965 | 781.70 | 3254.6 | 4.52 | 91.57 |
| 4 | wikitext2_line | 3.8515 | 0.000100 | 0.476 | N/A | 0.000 | 0.7413 | 791.31 | 3264.6 | 4.43 | 83.69 |
| 5 | wikitext2_line | 3.5379 | 0.000100 | 0.450 | N/A | 0.000 | 0.7978 | 788.22 | 3263.3 | 4.39 | 80.94 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.07, wikitext2_line Perplexity: 80.94
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.3937069729192935, "perplexity": 80.93990558124426}
