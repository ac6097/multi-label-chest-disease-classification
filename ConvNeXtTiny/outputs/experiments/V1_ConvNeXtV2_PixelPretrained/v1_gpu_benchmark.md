# V1 ConvNeXtV2 GPU Training Benchmark

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU; total VRAM: 8.00 GiB
- PyTorch: 2.10.0+cu128; CUDA: 12.8; timm: 1.0.30
- Each candidate includes real NAS DataLoader batches, H2D transfer, ConvNeXtV2 forward, Study Logit Mean, Standard BCE, backward, clipping, gradient accumulation, and AdamW step.
- Each candidate used 5 warmup updates and 20 measured updates.
- Gradient accumulation targets roughly 32 images/update where the discrete budget permits (for example 12×3=36 and 24×1=24); actual images, Studies, and effective images/update are recorded per candidate.
- Only candidates within 15% of the 32-image target, using both nominal and measured effective images/update, are eligible for selection. A 24×1 candidate is measured but excluded from selection because it changes the effective batch materially.
- Compute images/s sums CUDA-event timing around model/loss/backward and optimizer step. End-to-end images/s includes batch loading and transfer and is the selection metric.
- A candidate needs at least 10% reserved-memory headroom. Within 3% of the fastest feasible candidate, selection prefers more headroom.
- Selected: image_budget=32, grad_accumulation=1, checkpointing=True; 29.94 end-to-end images/s, 40.40 compute images/s, reserved peak 3096 MiB.

## Candidate results

| Budget | Accumulation | Effective images/update | Checkpointing | Status | Images/Studies/Subjects | Updates | E2E images/s | Compute images/s | Update P50/P95 sec | Update CV | Alloc/Reserved MiB | Mean GPU memory MiB | GPU util % | DataLoader wait sec | Error |
|---:|---:|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 | 16 | 30.2 | False | ok | 604/376/371 | 20 | 22.34 | 52.95 | 1.345/1.512 | 0.083 | 1388/1462 | 1588 | 25.2 | 15.50 |  |
| 4 | 8 | 28.4 | False | ok | 568/354/347 | 20 | 25.27 | 65.98 | 1.085/1.468 | 0.190 | 1382/1512 | 1797 | 41.7 | 13.75 |  |
| 8 | 4 | 30.1 | False | ok | 601/372/366 | 20 | 24.12 | 43.82 | 1.142/1.678 | 0.758 | 2251/2532 | 2701 | 32.4 | 11.09 |  |
| 12 | 3 | 34.8 | False | ok | 696/427/423 | 20 | 25.44 | 62.63 | 1.622/1.924 | 0.387 | 3123/3662 | 3736 | 30.1 | 16.13 |  |
| 16 | 2 | 31.0 | False | ok | 620/383/378 | 20 | 25.06 | 39.65 | 0.575/2.249 | 1.038 | 4008/4056 | 4109 | 38.4 | 9.00 |  |
| 24 | 1 | 23.4 | False | ok | 469/289/283 | 20 | 34.14 | 61.14 | 0.509/1.238 | 0.510 | 5614/7172 | 6229 | 29.9 | 5.97 |  |
| 32 | 1 | 31.6 | False | ok | 631/390/385 | 20 | 32.63 | 38.14 | 0.530/2.973 | 1.415 | 7374/7626 | 6918 | 36.5 | 2.68 |  |
| 32 | 1 | 31.6 | True | ok | 631/390/385 | 20 | 29.94 | 40.40 | 0.644/3.089 | 0.831 | 2689/3096 | 3040 | 49.9 | 5.33 |  |
| 12 | 3 | 34.8 | True | ok | 696/427/423 | 20 | 24.42 | 51.62 | 1.517/2.122 | 0.376 | 1293/1634 | 1880 | 40.5 | 14.90 |  |
