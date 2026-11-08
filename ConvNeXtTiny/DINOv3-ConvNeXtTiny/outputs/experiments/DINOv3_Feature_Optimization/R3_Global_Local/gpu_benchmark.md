# R3_Global_Local GPU throughput benchmark

The screen covered all planned image budgets with 4 workers, BF16, gradient checkpointing and channels_last. The refinement stage compared the worker count, checkpointing and memory format at the selected image budget. Every candidate used real Study batches, full preprocessing, Study Logit Mean, BCE, backward, optimizer step, and CUDA synchronization.

## Selection

| Phase | Budget | Accumulation | Workers | Checkpointing | Channels Last | Long mean images/s | CV | Peak reserved fraction |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Screen | 8 | 4 | 4 | True | True | 31.95 | 0.0078 | 0.133 |
| Refine | 8 | 4 | 4 | False | True | 33.19 | 0.0055 | 0.197 |

## Measurement details

- Screen candidate rows: 18.
- Refinement candidate rows: 21.
- Selected runtime: budget 8, accumulation 4, workers 4, checkpointing False, channels_last True.
- Benchmark timing reports end-to-end time separately from GPU compute time, includes actual NAS image loading and preprocessing, and synchronizes CUDA before stopping each timer.
- Formal epoch throughput is recorded in `train_log.csv` and can differ from short and repeated benchmark measurements.
- Test split was not read.
