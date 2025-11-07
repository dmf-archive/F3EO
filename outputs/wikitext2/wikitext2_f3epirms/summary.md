# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EPIRMS |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0005 |
| Epochs | 30 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Log(PI) | Time |
|-----|--|-----|-----|-----|-----|--------|------|
| 1 | 18.7607 | 10.6194 | 140500942.27 | 40922.50 | 0.000100 | 0.604 | 934.92s |
| 2 | 9.6015 | 9.4945 | 14786.35 | 13286.21 | 0.000100 | 9.016 | 935.27s |
| 3 | 8.7213 | 8.4926 | 6132.28 | 4878.71 | 0.000100 | 3.745 | 934.07s |
| 4 | 8.3267 | 8.1428 | 4132.90 | 3438.40 | 0.000100 | 1.986 | 932.31s |
| 5 | 8.0918 | 8.7199 | 3267.57 | 6123.51 | 0.000100 | 3.500 | 918.00s |
| 6 | 7.9715 | 7.8799 | 2897.12 | 2643.51 | 0.000100 | 1.444 | 899.99s |
| 7 | 7.8616 | 7.7732 | 2595.67 | 2376.15 | 0.000100 | -1.079 | 938.46s |
| 8 | 7.8007 | 7.8007 | 2442.33 | 2442.41 | 0.000100 | 1.306 | 937.00s |
| 9 | 7.7481 | 7.9477 | 2317.20 | 2829.09 | 0.000100 | 3.568 | 938.59s |
| 10 | 7.7106 | 7.8132 | 2231.96 | 2472.96 | 0.000100 | 1.899 | 939.93s |

## Performance Summary
- **Best Validation Perplexity**: 2376.15
- **Final Validation Perplexity**: 2472.96
- **Total Training Time**: 9327.03s
- **Average Epoch Time**: 930.85s

## Configuration Details
```toml
{
  "experiment": {
    "task": "wikitext2",
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "nano_gpt",
    "vocabulary_size": 40479,
    "embedding_size": 768,
    "sequence_length": 256,
    "num_heads": 12,
    "num_layers": 4
  },
  "data": {
    "batch_size": 8,
    "num_workers": 4,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "F3EPIRMS",
    "lr": 0.0001,
    "weight_decay": 0.0005,
    "alpha": 1.0,
    "gamma": 1.0,
    "beta2": 0.999,
    "meta_grad_clip_norm": 1.0,
    "orthogonalize": true
  },
  "train": {
    "epochs": 30,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
