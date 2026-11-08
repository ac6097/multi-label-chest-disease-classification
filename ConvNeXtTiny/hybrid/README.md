# Hybrid ConvNeXtV2–ViT experiment route

This route is isolated from the existing ConvNeXt-Tiny experiment outputs. It uses timm 1.0.30 with the verified ImageNet model tags `convnextv2_tiny.fcmae_ft_in1k` and `vit_small_patch16_224.augreg_in1k`; one shared checkpoint at `outputs/hybrid/shared/imagenet_initial.pth` is the only initialization source for H0/H1/H2. The ViT’s 14×14 patch position grid is resampled to 20×20 by timm `set_input_size`; the CLS position embedding is retained.

The dataset transform returns one shared RGB tensor in [0,1]. Each branch applies its own ImageNet normalization inside the model. H1/H2 read only Train image IDs and paths, with CSV label columns excluded from the self-supervised dataset reader. Train/Validation manifests record image, Study and Subject IDs; the audit checks image existence and Subject separation. Test CSV and images are not accessed by the training route.

## Execution order

Run these from `ConvNeXtTiny/`:

```powershell
python -m hybrid.audit_data
python -m hybrid.initialize_weights
python -m hybrid.smoke_test
python -m hybrid.benchmark
```

After the benchmark writes its measured choices to `configs/hybrid_train.yaml`, `configs/hybrid_h1_pretrain.yaml` and `configs/hybrid_h2_pretrain.yaml`, run:

```powershell
./run_hybrid_training.ps1
```

The training script runs H0, H1 pretraining and fine-tuning, then H2 pretraining and fine-tuning. Each step has an independent output directory and checkpoint; an existing run is never silently overwritten.

## Masked pretraining details

- H1 applies the same exact 60-of-100 random 32×32 pixel mask to both input branches and reconstructs only masked pixels against the augmented, unmasked [0,1] tensor.
- H1 uses ordinary dense PyTorch convolution on zero-filled inputs. It is not an implementation of the sparse-convolution FCMAE encoder; original hidden pixels are removed from the input, while visible-neighbor context can still pass through dense convolutions.
- H2 masks the Stage 3 map after it has been computed from the complete image, and masks ViT Block 6 patch tokens after the first six blocks. Detached representations are targets. This can retain pre-mask contextual information, as specified in the prompt.
- H1/H2 use the same train image order and seed. Their exact 60% spatial mask is keyed by Subject/Study/image ID, so mask positions stay matched even when the benchmark selects different microbatch sizes; both target a 32-image effective batch. H2 logs feature variance, branch/segment gradients, decoder output spread and parameter changes; detected collapse stops the run and writes diagnostics.
- Both methods export encoder-only weights for fine-tuning. Pixel/latent decoders remain only in self-supervised checkpoints.

## Method references

- ConvNeXt V2 and FCMAE: [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf), [arXiv HTML](https://arxiv.org/html/2301.00808v1).
- timm implementation and pretrained model registry: [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [ImageNet results registry](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv).
- The exact title “Hybrid ConvNeXtV2–ViT Architecture for Chest X-Ray Diagnosis (2026)” was not found in the available literature search. This route follows the detailed architecture and experimental contract supplied in `prompt_ConvNeXtV2&ViT.txt`; it does not claim to reproduce that unverified paper.
