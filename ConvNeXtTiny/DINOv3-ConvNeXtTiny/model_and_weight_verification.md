# DINOv3 ConvNeXt-Tiny 权重与环境核验

## 目标模型

- Hugging Face 仓库：[`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m)
- 官方实现仓库：[`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3)
- Hugging Face Model API 返回：仓库为 public，`gated=manual`，`library_name=transformers`，文件列表包含 `config.json` 与 `model.safetensors`。
- 模型 API revision：`10d30274b4d445111e2d5bf75ac93bbd94db274b`（2026-10-28 查询）。

## 本机访问结果

- Windows Python：`F:\Anaconda\python.exe`，Python 3.12.3，`huggingface_hub` 1.14.0。
- `HF_TOKEN`、`HF_HOME`、`HUGGINGFACE_HUB_CACHE` 环境变量均未设置。
- 默认 Hugging Face Hub 缓存中未发现该模型仓库的缓存目录。
- 未认证请求模型 `config.json` 返回 HTTP 401。
- 通过官方 `huggingface_hub.hf_hub_download(..., token=True)` 检查本机授权时，返回 `LocalTokenNotFoundError`：本机没有已登录的 Hugging Face Token。
- 使用者已在 Hugging Face 网页端获准访问模型，但本机 `hf auth login` 在服务端验证阶段发生 `SSLEOFError`；随后 `hf auth whoami` 仍返回 `Not logged in`，说明登录状态未写入。
- 后续匿名网络检查通过：Python `httpx` 请求 whoami API 返回预期 HTTP 401，模型 API 返回 HTTP 200；WinHTTP 及常见代理环境变量未配置代理。首次 SSL EOF 当前未复现。
- 因此尚未下载或读取 `model.safetensors`，没有计算权重 SHA-256，也没有验证网络结构的权重加载兼容性。

## 数据与边界核验

- 官方训练 CSV：`../cxr_first/train.csv`，存在。
- 官方验证 CSV：`../cxr_first/valid.csv`，存在。
- 未读取 Test 数据或 Test 标签。
- 本次尚未修改模型代码、训练配置或原始数据；未运行 Benchmark 或正式训练。
- 未使用随机初始化、V1/H1 权重、第三方权重或医学预训练权重替代指定 DINOv3 权重。

## 阻碍与恢复条件

指定仓库受 Hugging Face 人工授权门控。本机当前没有该仓库的授权凭据，因此按照实验方案停止在模型权重获取阶段。解除阻碍需要模型使用者先在 Hugging Face 模型页面申请并接受该模型的访问条件，再在本机安全地登录 Hugging Face。不要把访问令牌粘贴到聊天或提交到仓库。完成登录后可从本报告对应工作目录恢复核验和实验。

## 来源

- Hugging Face Model API：<https://huggingface.co/api/models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m>
- Hugging Face 模型卡：<https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m>
- Meta 官方实现：<https://github.com/facebookresearch/dinov3>
