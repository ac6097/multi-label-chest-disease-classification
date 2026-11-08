# DSAP-RPB Cosine20 / No Early Stopping LR Schedule Preview

- Updates per epoch (deterministic shuffled sampler): 1004–1010; total planned updates=40248
- Warmup updates: 2013
- Cosine end update: 20123
- Cosine steps: 18110
- Minimum LR: Backbone 5.000e-07; Head 1.000e-05; ratio=20.0
- Scheduler advances after each successful optimizer update; skipped GradScaler updates do not advance it.
- Cosine reaches the nonzero minimum on the final optimizer update of Epoch 20; Epoch 21–40 remain flat at that minimum.

| Epoch | Updates | Phase | Backbone LR start | Backbone LR last update | Head LR start | Head LR last update |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1005 | warmup | 4.96770989e-09 | 4.99254844e-06 | 9.93541977e-08 | 9.98509687e-05 |
| 2 | 1008 | warmup | 4.99751615e-06 | 1.00000000e-05 | 9.99503229e-05 | 2.00000000e-04 |
| 3 | 1006 | cosine | 1.00000000e-05 | 9.92798779e-06 | 2.00000000e-04 | 1.98559756e-04 |
| 4 | 1009 | cosine | 9.92784478e-06 | 9.71300735e-06 | 1.98556896e-04 | 1.94260147e-04 |
| 5 | 1004 | cosine | 9.71272519e-06 | 9.36368934e-06 | 1.94254504e-04 | 1.87273787e-04 |
| 6 | 1005 | cosine | 9.36327727e-06 | 8.88935841e-06 | 1.87265545e-04 | 1.77787168e-04 |
| 7 | 1007 | cosine | 8.88882881e-06 | 8.30341649e-06 | 1.77776576e-04 | 1.66068330e-04 |
| 8 | 1007 | cosine | 8.30278522e-06 | 7.62452422e-06 | 1.66055704e-04 | 1.52490484e-04 |
| 9 | 1005 | cosine | 7.62381050e-06 | 6.87489681e-06 | 1.52476210e-04 | 1.37497936e-04 |
| 10 | 1005 | cosine | 6.87412246e-06 | 6.07600102e-06 | 1.37482449e-04 | 1.21520020e-04 |
| 11 | 1006 | cosine | 6.07518952e-06 | 5.25123606e-06 | 1.21503790e-04 | 1.05024721e-04 |
| 12 | 1006 | cosine | 5.25041202e-06 | 4.42643355e-06 | 1.05008240e-04 | 8.85286711e-05 |
| 13 | 1006 | cosine | 4.42562201e-06 | 3.62665194e-06 | 8.85124401e-05 | 7.25330388e-05 |
| 14 | 1008 | cosine | 3.62587754e-06 | 2.87476212e-06 | 7.25175508e-05 | 5.74952425e-05 |
| 15 | 1006 | cosine | 2.87404854e-06 | 2.19658351e-06 | 5.74809709e-05 | 4.39316702e-05 |
| 16 | 1006 | cosine | 2.19595233e-06 | 1.61117119e-06 | 4.39190466e-05 | 3.22234238e-05 |
| 17 | 1007 | cosine | 1.61064159e-06 | 1.13589873e-06 | 3.22128317e-05 | 2.27179745e-05 |
| 18 | 1005 | cosine | 1.13548691e-06 | 7.86428737e-07 | 2.27097382e-05 | 1.57285747e-05 |
| 19 | 1005 | cosine | 7.86146980e-07 | 5.72298380e-07 | 1.57229396e-05 | 1.14459676e-05 |
| 20 | 1007 | cosine | 5.72155224e-07 | 5.00000000e-07 | 1.14431045e-05 | 1.00000000e-05 |
| 21 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 22 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 23 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 24 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 25 | 1005 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 26 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 27 | 1005 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 28 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 29 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 30 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 31 | 1004 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 32 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 33 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 34 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 35 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 36 | 1010 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 37 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 38 | 1007 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 39 | 1006 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |
| 40 | 1005 | low_lr_finetune | 5.00000000e-07 | 5.00000000e-07 | 1.00000000e-05 | 1.00000000e-05 |

## Previous Scheduler Readback

- Previous best Epoch/Macro-AUROC: 7 / 0.7858219278310556
- Previous last Epoch: 14
- Previous scheduler: custom LambdaLR (no explicit T_max/eta_min); effective cosine horizon after warmup=38190 optimizer updates, minimum LR factor=0.
- Previous completed run had 1004–1009 successful optimizer updates/epoch due to shuffled Study packing.
- Previous scheduler stepped on each successful optimizer update, after gradient accumulation; GradScaler-skipped updates did not advance it.

The previous DSAP-RPB run stopped after Epoch 14. This new run is a fresh ImageNet initialization and targets 40 epochs, so total optimizer updates differ.
