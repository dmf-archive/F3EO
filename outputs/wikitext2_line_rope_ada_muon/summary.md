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
    "name": "AdaMuon",
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
| 1 | wikitext2_line | 6.1269 | 0.000100 | 0.070 | N/A | 0.000 | 2.6640 | 676.76 | 2942.7 | 5.51 | 246.37 |
| 2 | wikitext2_line | 5.0399 | 0.000100 | 0.014 | N/A | 0.000 | 4.2870 | 674.52 | 2942.7 | 5.18 | 177.23 |
| 3 | wikitext2_line | 4.5148 | 0.000100 | 0.004 | N/A | 0.000 | 5.6155 | 674.87 | 2942.7 | 5.09 | 161.68 |
| 4 | wikitext2_line | 4.0356 | 0.000100 | 0.001 | N/A | 0.000 | 6.6268 | 675.71 | 2942.7 | 5.15 | 171.89 |
| 5 | wikitext2_line | 3.5569 | 0.000100 | 0.000 | N/A | 0.000 | 7.8954 | 674.65 | 2942.7 | 5.34 | 207.86 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.51, wikitext2_line Perplexity: 161.68
- **Final Validation Metrics**: wikitext2_line: {"loss": 5.336876203764731, "perplexity": 207.8623753554343}
