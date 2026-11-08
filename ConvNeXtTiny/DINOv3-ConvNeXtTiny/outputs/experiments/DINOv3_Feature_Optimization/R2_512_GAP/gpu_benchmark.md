# R2_512_GAP GPU throughput benchmark

The screen covered all planned image budgets with 4 workers, BF16, gradient checkpointing and channels_last. The refinement stage compared the worker count, checkpointing and memory format at the selected image budget. Every candidate used real Study batches, full preprocessing, Study Logit Mean, BCE, backward, optimizer step, and CUDA synchronization.

## Selection

| Phase | Budget | Accumulation | Workers | Checkpointing | Channels Last | Long mean images/s | CV | Peak reserved fraction |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Screen | 12 | 3 | 4 | True | True | 38.55 | 0.0096 | 0.170 |
| Refine | 12 | 3 | 4 | True | False | 38.68 | 0.0089 | 0.179 |

## Measurement details

- Screen candidate rows: 18.
- Refinement candidate rows: 21.
- Selected runtime: budget 12, accumulation 3, workers 4, checkpointing True, channels_last False.
- Benchmark timing reports end-to-end time separately from GPU compute time, includes actual NAS image loading and preprocessing, and synchronizes CUDA before stopping each timer.
- Formal epoch throughput is recorded in `train_log.csv` and can differ from short and repeated benchmark measurements.
- Test split was not read.
