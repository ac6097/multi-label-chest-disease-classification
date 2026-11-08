# DINOv3 ConvNeXt-Tiny 模型与权重核验

## 初始化来源

- Hugging Face：[facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m)
- 官方 Meta 实现：[facebook DINOv3](https://github.com/facebookresearch/dinov3)
- Revision: `10d30274b4d445111e2d5bf75ac93bbd94db274b`
- Meta ConvNeXt source blob: `6300de5a9af0b673e67ff7ae31258aa06c205080`
- 文件大小：111,299,216 bytes
- SHA-256：`bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`
- 权重张量：180；missing=0；unexpected=0
- 参数量：27,820,128 backbone + 7,690 classifier = 27,827,818
- 官方特征：768 维，GAP 后使用官方 LayerNorm；分类头前未额外重复池化或归一化。
- HF processor 默认尺寸为 224；本实验显式保留 320×320，并禁用默认 Resize/Crop。
- 预处理均值/标准差：`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- 权重来源为 Meta 通用网络图像自监督/蒸馏模型；未用医学预训练权重。
- 官方许可：[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)；模型使用者已在 Hugging Face 获得 gated 仓库访问权。

## 运行环境

- python: `3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:20:11) [MSC v.1938 64 bit (AMD64)]`
- platform: `win32`
- torch: `2.10.0+cu128`
- torchvision: `0.25.0+cu128`
- timm: `1.0.30`
- transformers_installed: `False`
- safetensors: `0.7.0`
- numpy: `2.3.3`
- pandas: `2.3.2`
- cuda_runtime: `12.8`
- cuda_available: `True`
- bf16_supported: `True`
- gpu_name: `NVIDIA GeForce RTX 4060 Laptop GPU`
- gpu_total_memory_bytes: `8585216000`
- gpu_compute_capability: `8.9`
- torch_cpu_threads: `24`
- cpu_logical_count: `32`

## 数据核验

- Train: 18,513 Studies / 30,344 images / 11,055 Subjects
- Valid: 859 Studies / 1,422 images / 228 Subjects
- Train 多图 Study: 10,006；单 Study 最大图像数：7
- Train 阳性 Study 数（1..10）: `{'Enlarged Cardiomediastinum': 622, 'Pneumothorax': 900, 'Consolidation': 942, 'Pneumonia': 1436, 'Edema': 2404, 'Cardiomegaly': 3885, 'Atelectasis': 3968, 'Lung Opacity': 4508, 'Pleural Effusion': 4654, 'No Finding': 6683}`
- 标签语义：`closed_world`；Test 数据未读取。
- Study sampler 保留完整 Study、每 epoch 每 Study 恰好一次；D0 与 D0-C1 共用相同采样器和随机种子。

## 权重加载结果

- 采用 Meta 官方 DINOv3 ConvNeXt-Tiny 运算结构，将 HF state tensors 严格加载；未安装或引入新运行时依赖。
- 模型实际张量形状、十分类 logits、320×320 输入、Study Logit Mean 和真实 Train BCE/对比梯度检查由后续 `smoke` 阶段记录。
