# MiniCertNet MNIST Training Report

Generated: 2026-11-07T14:20:52.715604+00:00

## Configuration

- Dataset: MNIST (`C:\Users\harsh\MiniCert\data`)
- Training samples: 60,000
- Test samples: 10,000
- Device: `cpu`
- Epochs: 5
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Elapsed time: 22.94 seconds

## Training history

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.537900 | 85.18% | 0.251451 | 92.81% |
| 2 | 0.228760 | 93.42% | 0.193104 | 94.20% |
| 3 | 0.173332 | 95.00% | 0.154590 | 95.34% |
| 4 | 0.140356 | 95.82% | 0.134961 | 95.90% |
| 5 | 0.120006 | 96.50% | 0.119279 | 96.44% |

## Test evaluation

- Test loss: 0.119279
- Test accuracy: 96.44%

## Checkpoint and prediction validation

- Checkpoint: `C:\Users\harsh\MiniCert\saved_model.pth`
- Strict state-dictionary reload into a fresh MiniCertNet: **Passed**
- Reloaded-model predictions matching the trained model: **10,000/10,000**
- Prediction test: **Passed**
- First test sample prediction: 7
- First test sample target: 7
