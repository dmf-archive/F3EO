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
    "name": "KFACRMSuon",
    "lr": 0.0001,
    "stat_decay": 0.95,
    "damping": 0.001,
    "TCov": 10,
    "ns_steps": 5,
    "weight_decay": 0.1
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
| 1 | wikitext2_line | 5.7973 | 0.000100 | 0.207 | N/A | 0.000 | 1.5728 | 477.97 | 2868.4 | 5.02 | 151.18 |
| 2 | wikitext2_line | 4.4849 | 0.000100 | 0.397 | N/A | 0.000 | 0.9245 | 500.64 | 2868.4 | 4.61 | 100.06 |
| 3 | wikitext2_line | 3.9316 | 0.000100 | 0.352 | N/A | 0.000 | 1.0448 | 515.63 | 2868.4 | 4.49 | 88.90 |
| 4 | wikitext2_line | 3.4984 | 0.000100 | 0.299 | N/A | 0.000 | 1.2060 | 493.03 | 2868.4 | 4.51 | 90.81 |
| 5 | wikitext2_line | 3.1061 | 0.000100 | 0.251 | N/A | 0.000 | 1.3836 | 491.42 | 2868.4 | 4.61 | 100.36 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.02, wikitext2_line Perplexity: 88.90
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.6087430840107935, "perplexity": 100.35792884313354}
