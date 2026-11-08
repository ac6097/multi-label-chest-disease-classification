# Hybrid ConvNeXtV2–ViT GPU Training Benchmark

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU; total VRAM: 8.00 GiB
- CUDA: 12.8; PyTorch: 2.10.0+cu128; timm: 1.0.30
- Timed each configuration with the real data loader, H2D transfer, forward, loss, backward, gradient accumulation, clipping, and optimizer update.
- Each configuration used 5 warmup updates and 20 measured updates; all workloads used a target effective batch of approximately 32 images.
- GPU compute time uses CUDA events around forward/backward, clipping and optimizer step; it excludes DataLoader waiting and H2D transfer. End-to-end throughput includes both and is used for selection. Allocated/reserved metrics come from CUDA; total GPU utilization and used memory were sampled with nvidia-smi.
- Gradient checkpointing was rechecked for the top two non-checkpointed throughput candidates per workload.

## h0_supervised

Selected: `image_budget=8`, `grad_accumulation=4`, `gradient_checkpointing=false`; 49.59 end-to-end images/s, 50.82 compute images/s, peak reserved 3222 MiB (39.4% of VRAM), 0.605s P50 / 0.661s P95 per update, 601 images / 372 Studies measured.
Highest measured rate was 49.59 images/s. The selection rule considers configurations within 3.0% of the highest rate, then prefers more reserved-memory headroom.

## h1_pixel

Selected: `image_budget=16`, `grad_accumulation=2`, `gradient_checkpointing=true`; 24.04 end-to-end images/s, 40.03 compute images/s, peak reserved 2264 MiB (27.7% of VRAM), 1.315s P50 / 1.629s P95 per update, 640 images / 633 Studies measured.
Highest measured rate was 24.04 images/s. The selection rule considers configurations within 3.0% of the highest rate, then prefers more reserved-memory headroom.

## h2_latent

Selected: `image_budget=8`, `grad_accumulation=4`, `gradient_checkpointing=true`; 23.05 end-to-end images/s, 40.43 compute images/s, peak reserved 1624 MiB (19.8% of VRAM), 1.387s P50 / 1.521s P95 per update, 640 images / 633 Studies measured.
Highest measured rate was 23.31 images/s. The selection rule considers configurations within 3.0% of the highest rate, then prefers more reserved-memory headroom.

## Candidate table

| Workload | Image budget | Grad accumulation | Effective batch target | Checkpointing | Status | Images / Studies / Subjects | Optimizer updates | End-to-end images/s | Compute images/s | P50/P95 update (s) | Peak alloc/reserved (MiB) | GPU util. | Mean total VRAM use | OOM/Error |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h0_supervised | 8 | 4 | 32 | False | ok | 601/372/366 | 20 | 49.59 | 50.82 | 0.605/0.661 | 3145/3222 | 57.5% | 38.0% |  |
| h0_supervised | 12 | 3 | 36 | False | ok | 696/427/423 | 20 | 49.51 | 49.91 | 0.707/0.729 | 4263/4546 | 51.0% | 52.9% |  |
| h0_supervised | 6 | 6 | 36 | False | ok | 656/405/400 | 20 | 43.62 | 50.93 | 0.715/0.862 | 2602/2680 | 59.4% | 32.1% |  |
| h0_supervised | 4 | 8 | 32 | False | ok | 568/354/347 | 20 | 42.81 | 49.61 | 0.667/0.763 | 2024/2114 | 79.8% | 28.0% |  |
| h0_supervised | 12 | 3 | 36 | True | ok | 696/427/423 | 20 | 40.88 | 41.16 | 0.856/0.882 | 1645/1918 | 72.7% | 24.9% |  |
| h0_supervised | 8 | 4 | 32 | True | ok | 601/372/366 | 20 | 38.37 | 38.63 | 0.739/0.962 | 1362/1530 | 73.0% | 20.9% |  |
| h0_supervised | 16 | 2 | 32 | False | ok | 620/383/378 | 20 | 34.40 | 34.61 | 0.631/0.980 | 5388/5626 | 49.7% | 64.9% |  |
| h0_supervised | 2 | 16 | 32 | False | ok | 604/376/371 | 20 | 18.97 | 37.26 | 1.609/1.757 | 2029/2164 | 33.1% | 28.3% |  |
| h0_supervised | 1 | 32 | 32 | False | ok | 1044/640/625 | 20 | 13.17 | 31.72 | 3.954/4.264 | 2031/2162 | 29.9% | 27.6% |  |
| h1_pixel | 16 | 2 | 32 | True | ok | 640/633/615 | 20 | 24.04 | 40.03 | 1.315/1.629 | 1966/2264 | 52.9% | 27.7% |  |
| h1_pixel | 12 | 3 | 36 | True | ok | 720/711/689 | 20 | 23.92 | 40.57 | 1.536/2.095 | 1671/1916 | 53.5% | 25.8% |  |
| h1_pixel | 12 | 3 | 36 | False | ok | 720/711/689 | 20 | 23.10 | 48.78 | 1.733/2.149 | 4368/4520 | 50.0% | 54.8% |  |
| h1_pixel | 16 | 2 | 32 | False | ok | 640/633/615 | 20 | 22.78 | 48.32 | 1.325/2.255 | 5537/5646 | 44.6% | 66.9% |  |
| h1_pixel | 4 | 8 | 32 | False | ok | 640/633/615 | 20 | 22.67 | 49.73 | 1.411/1.563 | 2068/2174 | 48.8% | 28.9% |  |
| h1_pixel | 8 | 4 | 32 | False | ok | 640/633/615 | 20 | 22.32 | 49.70 | 1.427/1.697 | 3225/3356 | 53.1% | 42.6% |  |
| h1_pixel | 6 | 6 | 36 | False | ok | 720/711/689 | 20 | 22.28 | 49.99 | 1.599/2.156 | 2660/2692 | 52.8% | 35.0% |  |
| h1_pixel | 2 | 16 | 32 | False | ok | 640/633/615 | 20 | 18.96 | 37.12 | 1.679/1.786 | 1495/1520 | 36.7% | 21.5% |  |
| h1_pixel | 1 | 32 | 32 | False | ok | 640/633/615 | 20 | 10.61 | 21.36 | 3.034/3.141 | 1203/1230 | 27.7% | 18.4% |  |
| h2_latent | 16 | 2 | 32 | False | ok | 640/633/615 | 20 | 23.31 | 48.24 | 1.221/2.259 | 5577/5826 | 45.4% | 67.7% |  |
| h2_latent | 8 | 4 | 32 | False | ok | 640/633/615 | 20 | 23.26 | 49.73 | 1.371/1.546 | 3257/3576 | 48.0% | 44.9% |  |
| h2_latent | 6 | 6 | 36 | False | ok | 720/711/689 | 20 | 23.11 | 49.10 | 1.511/2.166 | 2687/2774 | 42.8% | 35.2% |  |
| h2_latent | 8 | 4 | 32 | True | ok | 640/633/615 | 20 | 23.05 | 40.43 | 1.387/1.521 | 1412/1624 | 55.9% | 22.6% |  |
| h2_latent | 12 | 3 | 36 | False | ok | 720/711/689 | 20 | 22.90 | 48.98 | 1.744/2.129 | 4407/4540 | 43.1% | 55.2% |  |
| h2_latent | 4 | 8 | 32 | False | ok | 640/633/615 | 20 | 22.23 | 48.02 | 1.453/1.635 | 2094/2226 | 43.5% | 29.6% |  |
| h2_latent | 16 | 2 | 32 | True | ok | 640/633/615 | 20 | 22.04 | 39.83 | 1.361/2.236 | 1981/2396 | 58.0% | 30.5% |  |
| h2_latent | 2 | 16 | 32 | False | ok | 640/633/615 | 20 | 18.75 | 35.90 | 1.681/1.870 | 1519/1546 | 37.0% | 21.8% |  |
| h2_latent | 1 | 32 | 32 | False | ok | 640/633/615 | 20 | 11.50 | 21.09 | 2.759/2.965 | 1225/1242 | 28.3% | 18.6% |  |
