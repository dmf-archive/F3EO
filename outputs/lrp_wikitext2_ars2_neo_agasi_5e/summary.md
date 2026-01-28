# ARS-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2"
    ],
    "seed": 42,
    "device": "cuda",
    "epochs": 5
  },
  "model": {
    "type": "rope",
    "vocabulary_size": 40479,
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 4,
    "num_layers": 3,
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
    "name": "ARS2-Neo",
    "lr": 0.001,
    "betas": [
      0.9,
      0.95
    ],
    "rho": 0.3,
    "k": 5,
    "alpha": 0.1,
    "adaptive_sync": true,
    "asi_enabled": true,
    "adaptive_beta": 0.9,
    "adaptive_lambda": 1.0,
    "adaptive_gamma": 2.0
  },
  "pi": {
    "gamma": 1.0,
    "alpha": 1.0,
    "ema_beta": 0.9
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag alpha_t | Diag current_rho | Diag effective_k | Diag ema_gap | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Diag phi_std | Diag phi_t | Diag surrogate_gap | Diag threshold | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|--------------|------------------|------------------|--------------|----------------------------|----------------------------|--------------|------------|--------------------|----------------|-----------|-----------------|
| 1 | wikitext2 | 5.1527 | N/A | N/A | 0.001000 | 0.003 | N/A | 5.173 | 0.6209 | 601.14 | 3055.8 | 0.1067 | 0.3295 | 3.9088 | 0.0970 | 383.7050 | 31.2413 | 0.0616 | 0.0329 | 0.2113 | -0.0616 | 4.79 | 119.83 |
| 2 | wikitext2 | 4.1743 | N/A | N/A | 0.001000 | 0.006 | N/A | 4.232 | 0.5238 | 641.49 | 3437.0 | 0.1000 | 0.3177 | 4.0555 | 0.1014 | 527.6007 | 34.5224 | 0.0489 | -0.0003 | 0.0851 | -0.0489 | 4.54 | 93.40 |
| 3 | wikitext2 | 3.6810 | N/A | N/A | 0.001000 | 0.009 | N/A | 3.766 | 0.5258 | 618.50 | 3308.2 | 0.1066 | 0.3230 | 4.1068 | 0.1229 | 626.5234 | 37.5156 | 0.0274 | 0.0327 | 0.1346 | -0.0274 | 4.52 | 92.18 |
| 4 | wikitext2 | 3.2881 | N/A | N/A | 0.001000 | 0.012 | N/A | 3.404 | 0.5470 | 635.74 | 3436.9 | 0.1000 | 0.3146 | 4.1537 | 0.1486 | 701.8258 | 40.3735 | 0.0238 | -0.0109 | 0.1545 | -0.0238 | 4.57 | 96.68 |
| 5 | wikitext2 | 2.9348 | N/A | N/A | 0.001000 | 0.015 | N/A | 3.076 | 0.5782 | 634.03 | 3437.5 | 0.1023 | 0.2915 | 4.1214 | 0.1596 | 760.6528 | 43.0387 | 0.0224 | 0.0115 | 0.1604 | -0.0224 | 4.71 | 110.56 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 4.79, wikitext2 Perplexity: 92.18
- **Final Validation Metrics**: wikitext2: {"loss": 4.705525375124234, "perplexity": 110.55635325013901}
