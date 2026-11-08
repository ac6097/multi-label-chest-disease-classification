# DINOv3 泛化瓶颈诊断：G0 + G1 综合报告

本轮只做推理一致性与泛化诊断，未训练任何新模型，未读取 Test 标签。

## 1. 已排除的推理问题

- G0 gate passed = **True**。
- 研究验证入口与提交包入口在相同 859 个 Validation Study 上使用同一 checkpoint、同一预处理（ResizePad/Grayscale3/Normalize）、同一类别顺序、同一 Study Logit Mean 与 sigmoid 顺序；提交包代码与研究代码文件哈希一致，包内权重与 R3 `best.pth` 的 `model_state` 逐位一致。
- R3 的 Validation 指标可复现：历史保存 0.804653 vs 重放 0.804516（差异在 BF16/批次形状数值噪声范围内）。
- Study 集合/顺序、每 Study 图像集合、图像存在性、标签顺序均对齐。

## 2. 仍未排除的问题

- R3 提交包的 `channels_last=false` 与训练 runtime 的 `true` 不同；该配置差异只造成数值级差异（已量化），但正式提交实现保留了这一非训练配置。
- R5 历史保存 `val_best.csv` 与当前代码重放存在已知数值差异（最大概率差约 0.009），该差异与 R3（0.0075）同属 BF16 数值可复现性限制（训练时 GPU 内存/算法选择环境不同），不是逻辑错误；R7 的同类重放可复现到 7e-8。
- 官方 Test 0.7510 与 Validation 0.804653 的落差不属于推理实现问题，但本轮没有独立 Test 标签，无法分解为分布偏移/标签噪声/模型误差。

## 3. 训练集与验证集的真正差距

- R3：Train Macro-AUROC 0.903671 vs Validation 0.804516，差 +0.099156。
- R5：Train Macro-AUROC 0.903096 vs Validation 0.803469，差 +0.099627。
- 这是同一确定性无增强协议下的 Train/Valid 差距；Train 评价属于训练内评价，不能直接当作泛化估计，也不能单独归因于微调过度。

## 4. 主要瓶颈疾病

逐疾病 Train/Valid（同一确定性无增强协议）：

| Disease | R3 train | R3 valid | R3 gap | R5 train | R5 valid | R5 gap | Valid 阳性 Study |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.8389 | 0.7430 | +0.0959 | 0.8388 | 0.7079 | +0.1309 | 24 |
| Pneumothorax | 0.9396 | 0.8527 | +0.0870 | 0.9409 | 0.8552 | +0.0857 | 37 |
| Consolidation | 0.8878 | 0.7666 | +0.1212 | 0.8872 | 0.7885 | +0.0987 | 46 |
| Pneumonia | 0.8642 | 0.7228 | +0.1414 | 0.8656 | 0.7242 | +0.1414 | 68 |
| Edema | 0.9394 | 0.8816 | +0.0578 | 0.9361 | 0.8854 | +0.0508 | 115 |
| Cardiomegaly | 0.9036 | 0.7777 | +0.1259 | 0.9026 | 0.7769 | +0.1257 | 177 |
| Atelectasis | 0.8929 | 0.7951 | +0.0978 | 0.8923 | 0.7926 | +0.0997 | 193 |
| Lung Opacity | 0.8479 | 0.7275 | +0.1204 | 0.8460 | 0.7233 | +0.1227 | 214 |
| Pleural Effusion | 0.9650 | 0.9084 | +0.0566 | 0.9633 | 0.9088 | +0.0546 | 260 |
| No Finding | 0.9575 | 0.8700 | +0.0875 | 0.9582 | 0.8720 | +0.0862 | 306 |

- 泛化差距最大：Pneumonia（R3 +0.1414 / R5 +0.1414）、Cardiomegaly（+0.1259 / +0.1257）、Consolidation（+0.1212 / +0.0987）、Lung Opacity（+0.1204 / +0.1227）、Atelectasis（+0.0978 / +0.0997）、Enlarged Cardiomediastinum（+0.0959 / +0.1309）。
- 即使在 Train 上 AUROC 也偏低的是 Enlarged Cardiomediastinum（R3 0.8389）、Lung Opacity（0.8479）、Pneumonia（0.8642），与 Validation 阳性 Study 数很少（24 / 214 / 68）叠加，这些类别的估计噪声最大。
- 注意：Pneumonia/Cardiomegaly 等类别的 Valid 阳性数少（68/177），其 AUROC 的抽样不确定性远大于 Pleural Effusion（260）与 No Finding（306）。

## 4b. 单图/多图 Study 与聚合（Validation）

| Group | Study 组 | Study 数 | Subject 数 | Macro-AUROC | Macro-AUPRC | Study BCE |
|---|---|---:|---:|---:|---:|---:|
| R3 | single | 391 | 94 | 0.7282 | 0.3916 | 0.3961 |
| R3 | multi | 468 | 198 | 0.8505 | 0.4381 | 0.2418 |
| R3 | two | 380 | 176 | 0.8563 | 0.4503 | 0.2384 |
| R3 | three_plus | 88 | 60 | 0.7970 | 0.4417 | 0.2565 |
| R5 | single | 391 | 94 | 0.7279 | 0.4025 | 0.3924 |
| R5 | multi | 468 | 198 | 0.8500 | 0.4476 | 0.2434 |
| R5 | two | 380 | 176 | 0.8540 | 0.4608 | 0.2411 |
| R5 | three_plus | 88 | 60 | 0.8082 | 0.4430 | 0.2531 |

- 单图 Study 的 Macro-AUROC（R3 0.7282 / R5 0.7279）明显低于多图 Study；但两组疾病构成与 Subject 组成不同，不能仅据此判定聚合算法优劣。
- 预先指定的固定权重聚合对照（R5，468 个多图 Study，Logit Mean vs Probability Mean）：Δ Macro-AUROC = -0.003661，95% CI [-0.00991220623254776, 0.00034100562334060143]；点估计不支持改用 Probability Mean，且 CI 跨 0，因此维持 Logit Mean。

## 5. 参数与特征偏移

- 冻结的 Stem/Stage1/Stage2 参数逐位未变（R3/R5 均通过 frozen check）。
- Stage4 相对参数漂移：R3 0.007984，R5 0.008006；Stage3 较小（见 G1-D 表）。
- Stage4 GAP 原始特征与预训练余弦（Validation）：R3 -0.0126，R5 0.1340；分类器输入(final)特征余弦：R3 0.0704，R5 0.2087。Stage3 GAP 余弦≈0.998，表明 Stage3 原始特征基本保留，漂移主要集中在 Stage4 与最终投影。
- 同 run 内 Best(ep6) vs Last(ep9) 对照：Stage4 原始 GAP 余弦 R3 0.6099、R5 0.5752；说明 raw Stage4 GAP 对微小参数变化高度敏感，其绝对余弦不宜单独作为过拟合证据。
- 漂移确实存在且 R3/R5 现象一致，但漂移本身不等于泛化退化；它只说明微调显著改变了任务相关表示。

## 6. 是否支持降低 Stage3/4 学习率

- 决策：**建议在用户确认后执行 G2**。
- 建议在用户确认后启动 G2（1e-5 vs 3e-6 配对种子实验）。依据：Train−Valid Macro-AUROC 差距 R3=+0.0992/R5=+0.0996；Stage4 参数相对漂移 R3=0.0080/R5=0.0080；分类器输入(final)特征余弦 R3=0.0704/R5=0.2087（原始 Stage4 GAP 余弦 R3=-0.0126/R5=0.1340）。这些是可检验“微调强度影响泛化”的合理依据，但不构成因果证明。

## 7. 三个配对 Seed 是否出现方向一致的改善

- G2 未执行（按门控要求等待用户确认），因此该问题目前**无法回答**。

## 8. 下一阶段优先方向

- 若用户批准 G2：先做固定矩阵（1e-5 vs 3e-6 × seeds 42/2026/3407）的配对实验，并同时报告参数漂移曲线与 Train/Valid gap 变化。
- 若不批准 G2 或希望先补证据：优先检查 Validation 与 Test 的分布/来源差异、困难疾病的标签一致性与特征可分性，以及多图 Study 的聚合方式（本轮 Probability Mean − Logit Mean 的 Δ Macro-AUROC = -0.003661，95% CI [-0.00991220623254776, 0.00034100562334060143]）。

## 9. 官方 Test 0.7510 落差能解释到什么程度

- 已排除：提交入口与验证入口的推理实现不一致、预处理/标签顺序/Study 聚合错误。
- 仍未解释：Validation 0.804653 → Test 0.7510 的约 0.0537 绝对落差。G0 证明该落差不是推理管线 bug；G1 证明模型在 Train 上显著高于 Validation，但 Validation 到 Test 的偏移缺少独立标签而无法量化。

## 10. 因缺少独立测试数据而无法证实的推论

- 无法确认 Test 分布偏移的具体幅度与方向（来源/设备/视角差异）。
- 无法确认降低 Stage3/4 学习率会改善 Test 泛化；提升 Validation 也不等于提升 Test。
- 无法确认参数/特征漂移与 Test 性能之间的因果关系。
- 无法区分验证集过拟合（选模偏差）与真实分布差异各自贡献了多少。

## 关键瓶颈假设（证据分级）

- **已确认事实**：Train−Valid Macro-AUROC 差距约 +0.099（R3/R5 一致）；Stem/Stage1/Stage2 逐位未变；Stage3 参数相对漂移 0.0084、Stage4 0.0080；Stage3 GAP 特征余弦 0.998，Stage4/final 特征显著漂移；单图 Study 指标低于多图；Logit Mean 优于 Probability Mean 的点估计（CI 跨 0）。
- **支持“泛化/分布问题优先于推理或聚合问题”的证据**：G0 排除了推理实现差异；R7–R10 的新融合/注意力结构均未提升；多图聚合方式不是瓶颈（固定权重对照无优势）；泛化差距在两个不同结构上几乎相同（+0.0992/+0.0996），更像数据级现象。
- **反对“微调严重破坏预训练表征”的证据**：冻结层逐位未变；Stage3/4 参数相对漂移仅约 0.8%；Stage3 原始特征基本保留；因此不能把差距直接归因于 Backbone 被过度改变。
- **尚不能判断**：官方 Test 0.7510 与 Validation 0.8047 的落差中，验证集选模偏差、Test 分布偏移、标签噪声各占多少；以及降低 Stage3/4 学习率能否改善跨Subject 泛化。
- **当前最可能瓶颈**：数据层面（Train/Validation/Test 的 Subject/来源/标签分布差异与困难疾病的有效样本量），其次才是微调强度；两者都需要独立 Test 标签或分布元数据才能进一步分离。
- **分布差异事实**：Train 11,055 Subject、平均 1.68 Study/Subject（最大 21）；Validation 仅 228 Subject、平均 3.77 Study/Subject（最大 89），且两者 Subject 不重叠。验证集指标对少数 Subject 的权重明显更高，这既影响 Subject-cluster 不确定性，也提示 Validation 的 Subject 组成与 Train 不同，不能直接代表 Test 分布。

## 产物索引

- G0：`G0_Inference_Consistency/`（inference_protocol_audit.md, study_alignment.csv, preprocessing_comparison.json, logit_probability_difference.csv, per_class_metric_comparison.csv, g0_final_report.md 等）。
- G1：`G1_R3_R5_Generalization/`、`G1_Parameter_Feature_Drift/`、`G1_Study_Aggregation_Check/` 下的规定 CSV/PNG 与报告。
- 决策：`G1_R3_R5_Generalization/g2_go_no_go_decision.md`。

