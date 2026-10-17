# 分析总结与下一步实验建议

## 当前事实

1. `cxr_first` 包含训练、验证、测试图像；按 `(Subject_id, Study_id)` 统计分别为 18,513、859、1,414 个 Study。
2. CSV 的 `Predict_class` 为分号分隔的阳性类别 ID；同一 Study 的多张图标签一致。用户确认训练集和验证集中未列出的类别为阴性。训练/验证没有空标签行；测试标签不参与训练、估权或预测。
3. 两条路线都按 Study 加载数据，已把 `label_semantics` 设置为 `closed_world`。训练入口使用标准 BCE 默认配置；Masked BCE 保留未来处理未知/不确定元素的能力，Weighted BCE 候选从 train Study 计数计算。
4. 训练/验证类别计数和候选权重见 `label_distribution.md`、`bce_asl_analysis.md` 及 `tables/`。指标必须在 Study 级计算；验证集每类当前均有正负样本。

## 启动顺序

1. 分别进入 `DenseNet121/` 或 `ConvNeXtTiny/`，安装对应 `requirements.txt`，检查 PyTorch 能识别 GPU，然后执行 `python train.py`。两条路线共享只读的 `cxr_first/`，分别输出 checkpoint 和日志。
2. 首轮保持标准 BCE，先得到每条路线的 Study 级基线。随后逐项对照 Weighted BCE（建议 sqrt 与 cap=5）和 ASL，不同时改变其他实验因素。
3. 训练完成后使用各路线 `evaluate.py`、`predict.py` 和提交 CSV 校验脚本；按验证 Macro-AUROC 选择，并记录逐类 AUROC/AUPRC、样本数、耗时与显存。
4. 当前没有执行模型训练，因此没有任何模型 AUC 或性能提升结论。融合方法、ASL效果、模型延迟和 T4 的100ms计时口径仍需通过实验或官方说明确认。
