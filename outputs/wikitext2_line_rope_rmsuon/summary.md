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
    "name": "RMSuon",
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
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2_line | 5.4850 | 0.000100 | 0.155 | N/A | 0.000 | 1.8643 | 731.86 | 2943.3 | 4.99 | 146.52 |
| 2 | wikitext2_line | 4.4559 | 0.000100 | 0.143 | N/A | 0.000 | 1.9437 | 746.74 | 2943.3 | 4.67 | 106.24 |
| 3 | wikitext2_line | 3.9066 | 0.000100 | 0.091 | N/A | 0.000 | 2.3947 | 736.76 | 2943.3 | 4.60 | 99.07 |
| 4 | wikitext2_line | 3.4010 | 0.000100 | 0.048 | N/A | 0.000 | 3.0299 | 727.79 | 2943.3 | 4.69 | 108.74 |
| 5 | wikitext2_line | 2.8995 | 0.000100 | 0.024 | N/A | 0.000 | 3.7304 | 730.23 | 2943.3 | 4.90 | 134.11 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 4.99, wikitext2_line Perplexity: 99.07
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.898692753777575, "perplexity": 134.11434457116027}
