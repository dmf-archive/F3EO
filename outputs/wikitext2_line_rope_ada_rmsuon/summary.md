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
    "name": "AdaRMSuon",
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
| 1 | wikitext2_line | 5.3029 | 0.000100 | 0.458 | N/A | 0.000 | 0.7805 | 439.78 | 2944.3 | 4.90 | 133.68 |
| 2 | wikitext2_line | 4.4170 | 0.000100 | 0.488 | N/A | 0.000 | 0.7165 | 613.29 | 2944.3 | 4.59 | 98.13 |
| 3 | wikitext2_line | 3.9848 | 0.000100 | 0.464 | N/A | 0.000 | 0.7688 | 552.22 | 2944.3 | 4.46 | 86.54 |
| 4 | wikitext2_line | 3.6256 | 0.000100 | 0.432 | N/A | 0.000 | 0.8383 | 669.65 | 2944.3 | 4.43 | 83.88 |
| 5 | wikitext2_line | 3.2842 | 0.000100 | 0.398 | N/A | 0.000 | 0.9219 | 678.20 | 2944.3 | 4.47 | 87.61 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 4.90, wikitext2_line Perplexity: 83.88
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.472895154312475, "perplexity": 87.6100006688325}
