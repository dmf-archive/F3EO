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
    "name": "RMSuon",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.0,
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
| 1 | wikitext2 | 6.2517 | 0.000100 | 0.164 | N/A | 0.000 | 1.8074 | 437.67 | 2943.3 | 5.69 | 296.26 |
| 2 | wikitext2 | 5.0769 | 0.000100 | 0.123 | N/A | 0.000 | 2.0944 | 482.74 | 2943.3 | 5.33 | 205.57 |
| 3 | wikitext2 | 4.4130 | 0.000100 | 0.071 | N/A | 0.000 | 2.6453 | 646.50 | 2943.3 | 5.26 | 191.85 |
| 4 | wikitext2 | 3.7911 | 0.000100 | 0.034 | N/A | 0.000 | 3.3737 | 633.64 | 2943.3 | 5.37 | 214.92 |
| 5 | wikitext2 | 3.1760 | 0.000100 | 0.016 | N/A | 0.000 | 4.1232 | 625.18 | 2943.3 | 5.64 | 281.74 |
| 6 | wikitext2 | 2.5847 | 0.000100 | 0.008 | N/A | 0.000 | 4.7711 | 615.52 | 2943.3 | 6.02 | 410.42 |
| 7 | wikitext2 | 2.0450 | 0.000100 | 0.005 | N/A | 0.000 | 5.2348 | 616.05 | 2943.3 | 6.48 | 651.56 |
| 8 | wikitext2 | 1.5737 | 0.000100 | 0.004 | N/A | 0.000 | 5.5113 | 617.95 | 2943.3 | 6.94 | 1037.25 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 6.94, wikitext2 Perplexity: 191.85
- **Final Validation Metrics**: wikitext2: {"loss": 6.944331189681744, "perplexity": 1037.2530396331822}
