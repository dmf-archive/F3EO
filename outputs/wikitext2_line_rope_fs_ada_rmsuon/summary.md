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
    "name": "FS_AdaRMSuon",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "rho": 0.05
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
| 1 | wikitext2_line | 5.8358 | 0.000100 | 0.049 | N/A | 0.000 | 3.0183 | 1279.86 | 3332.9 | 5.27 | 193.75 |
| 2 | wikitext2_line | 4.8326 | 0.000100 | 0.077 | N/A | 0.000 | 2.5656 | 1370.94 | 3561.3 | 4.88 | 131.21 |
| 3 | wikitext2_line | 4.3902 | 0.000100 | 0.068 | N/A | 0.000 | 2.6905 | 1371.69 | 3560.2 | 4.71 | 111.06 |
| 4 | wikitext2_line | 4.0388 | 0.000100 | 0.048 | N/A | 0.000 | 3.0426 | 1375.86 | 3562.2 | 4.65 | 104.79 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.27, wikitext2_line Perplexity: 104.79
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.6519676749386, "perplexity": 104.79097742822731}
