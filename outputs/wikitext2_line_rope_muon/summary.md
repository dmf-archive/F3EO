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
    "name": "Muon",
    "lr": 0.0001,
    "weight_decay": 0.1,
    "momentum": 0.95,
    "adam_lr": 0.001,
    "adam_weight_decay": 0.1,
    "adam_betas": [
      0.9,
      0.95
    ]
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
| 1 | wikitext2_line | 5.7051 | 0.000100 | 0.373 | N/A | 0.000 | 0.9872 | 727.20 | 2868.3 | 5.45 | 233.30 |
| 2 | wikitext2_line | 5.0654 | 0.000100 | 0.422 | N/A | 0.000 | 0.8628 | 714.20 | 2868.3 | 5.23 | 186.67 |
| 3 | wikitext2_line | 4.8151 | 0.000100 | 0.384 | N/A | 0.000 | 0.9570 | 709.49 | 2868.3 | 5.13 | 169.55 |
| 4 | wikitext2_line | 4.6797 | 0.000100 | 0.355 | N/A | 0.000 | 1.0364 | 719.76 | 2868.3 | 5.10 | 164.37 |
| 5 | wikitext2_line | 4.5879 | 0.000100 | 0.331 | N/A | 0.000 | 1.1046 | 729.37 | 2868.3 | 5.08 | 161.09 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.45, wikitext2_line Perplexity: 161.09
- **Final Validation Metrics**: wikitext2_line: {"loss": 5.081934626422711, "perplexity": 161.08539473032334}
