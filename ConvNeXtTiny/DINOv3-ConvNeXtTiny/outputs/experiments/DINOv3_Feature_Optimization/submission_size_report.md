# DINOv3 feature optimization submission package sizes

- Strict limit: 500,000,000 bytes.
- Each package is FP32, stores only inference weights and required code, loads strictly, and validates a multi-image Validation Study through the standalone entrypoint.
- The probability check uses a 1e-2 absolute tolerance for BF16 batch-shape numerical differences; actual maximum absolute error is reported per package.
- Test split was not read.

| Group | Backbone/head | FP32 weights bytes | ZIP bytes | Uncompressed package bytes | Passes limit | Validation images | Max probability abs error / tolerance | Mean inference ms/Study | Peak GPU GiB |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| R1 | DINOv3 ConvNeXt-Tiny / gap | 111,336,888 | 103,621,335 | 111,524,288 | True | 2 | 0.00025641918 / 0.0005 | 395.72 | 0.179 |
| R2 | DINOv3 ConvNeXt-Tiny / gap | 111,336,888 | 103,621,827 | 111,524,288 | True | 2 | 0.00041034818 / 0.0005 | 380.70 | 0.184 |
| R3 | DINOv3 ConvNeXt-Tiny / global_local | 111,352,548 | 103,635,810 | 111,540,162 | True | 2 | 0.00048194081 / 0.0005 | 400.01 | 0.188 |
| R4 | DINOv3 ConvNeXt-Small / gap | 197,892,832 | 184,022,535 | 198,080,312 | True | 2 | 0.0049481988 / 0.01 | 404.05 | 0.321 |
