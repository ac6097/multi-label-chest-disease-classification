# DINOv3 ConvNeXt-Tiny R7 inference package

`R7Predictor` loads the packaged Best checkpoint and returns ten Study-level probabilities. Pass the PIL images or file paths belonging to one Study to `predict_study_images` or `predict_study_paths`. Study aggregation is the arithmetic mean of raw image logits followed by sigmoid.

The package contains the Best model weights only; optimizer and scheduler states are excluded. The model was initialized from the pinned general-vision DINOv3 ConvNeXt-Tiny checkpoint identified in `model_config.json`.
