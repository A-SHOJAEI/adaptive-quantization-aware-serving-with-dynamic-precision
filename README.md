# Adaptive Quantization-Aware Serving with Dynamic Precision

A production ML serving system that dynamically adjusts quantization precision (INT8/FP16/FP32) based on real-time latency budgets and accuracy requirements. Combines quantization-aware training with runtime precision switching controlled by a lightweight accuracy predictor network to optimize the throughput-accuracy Pareto frontier.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Train with quantization-aware training and accuracy predictor
python scripts/train.py --config configs/default.yaml

# Evaluate precision switching performance
python scripts/evaluate.py --checkpoint models/best_model.pth

# Run inference with dynamic precision
python scripts/predict.py --input data/sample.json --latency-budget 10
```

## Key Features

- Quantization-aware training with INT8/FP16/FP32 support
- Runtime precision switching with <0.5ms overhead
- Lightweight accuracy predictor network for precision selection
- Triton Inference Server integration
- MLflow experiment tracking and model versioning

## Results

Evaluated on CIFAR-10 (50,000 train / 10,000 test images, 50 epochs, seed=42):

**Training Performance:**

| Metric | Value |
|--------|-------|
| Train Accuracy | 92.03% |
| Val Accuracy | 84.40% |
| Best Val Loss | 1.146 |

**Test Set Evaluation:**

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.63% |
| Precision (macro) | 86.62% |
| Recall (macro) | 86.63% |
| F1 Score (macro) | 86.61% |
| Avg Inference Time | 0.38 ms/sample |
| Throughput | 2,662 samples/sec |

**Per-Class Test F1 Scores:**

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| airplane | 87.24% | 89.60% | 88.41% |
| automobile | 92.04% | 93.70% | 92.86% |
| bird | 83.71% | 84.80% | 84.25% |
| cat | 71.61% | 72.90% | 72.25% |
| deer | 87.09% | 87.70% | 87.39% |
| dog | 79.83% | 75.20% | 77.45% |
| frog | 91.16% | 90.70% | 90.93% |
| horse | 90.60% | 90.60% | 90.60% |
| ship | 92.25% | 91.70% | 91.98% |
| truck | 90.67% | 89.40% | 90.03% |

## Architecture

The system consists of three components:
1. Quantization-aware trained models (INT8/FP16/FP32 variants)
2. Accuracy predictor network (MLP predicting per-sample precision needs)
3. Dynamic serving runtime with precision switching logic

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
