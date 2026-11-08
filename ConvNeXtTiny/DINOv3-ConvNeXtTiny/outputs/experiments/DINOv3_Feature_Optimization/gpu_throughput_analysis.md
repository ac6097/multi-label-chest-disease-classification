# DINOv3 feature optimization throughput analysis

Benchmark long-measurement throughput and formal epoch throughput are reported separately. Long candidates include CUDA synchronization and actual Study batches; formal epochs include every official Train Study and all of its images.

| Group | Resolution | Budget | Accum | Workers | Checkpointing | Channels Last | Long mean images/s | Formal mean images/s | Mean loader wait (s/epoch) | Peak reserved GiB |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| R1_448_GAP | 448 | 12 | 3 | 4 | False | False | 39.31 | 22.32 | 997.1 | 1.69 |
| R2_512_GAP | 512 | 12 | 3 | 4 | True | False | 38.68 | 22.99 | 795.8 | 1.34 |
| R3_Global_Local | 512 | 8 | 4 | 4 | False | True | 33.19 | 21.45 | 1048.6 | 1.71 |
| R4_ConvNeXt_Small | 512 | 12 | 3 | 4 | False | True | 37.81 | 24.17 | 643.6 | 5.51 |

Per-resolution candidate timings and the reason for the selected runtime are in each group's `gpu_benchmark.md`.
