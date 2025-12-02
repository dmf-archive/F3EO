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
| 1 | wikitext2 | 6.0728 | 0.000100 | 0.433 | N/A | 0.000 | 0.8366 | 416.98 | 2944.2 | 5.60 | 270.49 |
| 2 | wikitext2 | 5.0642 | 0.000100 | 0.446 | N/A | 0.000 | 0.8065 | 423.85 | 2944.2 | 5.24 | 188.04 |
| 3 | wikitext2 | 4.5467 | 0.000100 | 0.423 | N/A | 0.000 | 0.8599 | 427.98 | 2944.2 | 5.08 | 161.20 |
| 4 | wikitext2 | 4.1137 | 0.000100 | 0.394 | N/A | 0.000 | 0.9302 | 423.12 | 2944.2 | 5.03 | 152.40 |
| 5 | wikitext2 | 3.7122 | 0.000100 | 0.363 | N/A | 0.000 | 1.0140 | 422.22 | 2944.2 | 5.06 | 157.27 |
| 6 | wikitext2 | 3.3177 | 0.000100 | 0.330 | N/A | 0.000 | 1.1073 | 423.34 | 2944.2 | 5.15 | 172.77 |
| 7 | wikitext2 | 2.9238 | 0.000100 | 0.300 | N/A | 0.000 | 1.2052 | 423.58 | 2944.2 | 5.32 | 204.73 |
| 8 | wikitext2 | 2.5348 | 0.000100 | 0.272 | N/A | 0.000 | 1.3002 | 425.17 | 2944.2 | 5.54 | 253.80 |
| 9 | wikitext2 | 2.1564 | 0.000100 | 0.251 | N/A | 0.000 | 1.3822 | 424.40 | 2944.2 | 5.82 | 335.34 |
| 10 | wikitext2 | 1.8065 | 0.000100 | 0.235 | N/A | 0.000 | 1.4469 | 424.84 | 2944.2 | 6.11 | 451.77 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 6.11, wikitext2 Perplexity: 152.40
- **Final Validation Metrics**: wikitext2: {"loss": 6.113165427898538, "perplexity": 451.7664884649357}
