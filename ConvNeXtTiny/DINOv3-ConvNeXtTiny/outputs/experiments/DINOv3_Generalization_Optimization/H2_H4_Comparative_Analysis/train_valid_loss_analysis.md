# H2-B / H2-C / H4 unified Train/Validation loss analysis

## Method

- 完整官方 Train/Validation，`model.eval()`，确定性无增强，Study-level 原始 logit，未加权 BCE，无单图权重、无正则项、无阈值。
- 每类 BCE 使用全部 Study；总体 BCE = 十类 BCE 均值；程序断言二者一致。
- Best/Last 均完成全量 Train/Validation 推理；训练批次 BCE 与 Validation 确定性 BCE 明确区分。

## Overall loss

| Experiment | Train BCE | Valid BCE | BCE Gap |
|---|---:|---:|---:|
| H0 | 0.236135 | 0.311185 | +0.075051 |
| H2 | 0.234986 | 0.311098 | +0.076112 |
| H2B | 0.234647 | 0.311292 | +0.076645 |
| H2C | 0.194044 | 0.331139 | +0.137095 |
| H4 | 0.239685 | 0.313042 | +0.073357 |

## Ten-class Train/Valid BCE (Best)

| Disease | H0 Train | H2 Train | H2B Train | H2C Train | H4 Train | H0 Valid | H2 Valid | H2B Valid | H2C Valid | H4 Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.125810 | 0.122395 | 0.123427 | 0.103686 | 0.126624 | 0.124876 | 0.120108 | 0.122149 | 0.119028 | 0.124092 |
| Pneumothorax | 0.108250 | 0.104915 | 0.105982 | 0.082331 | 0.115334 | 0.141200 | 0.138153 | 0.140751 | 0.144159 | 0.143488 |
| Consolidation | 0.150230 | 0.149358 | 0.149100 | 0.127561 | 0.150046 | 0.178681 | 0.178796 | 0.178520 | 0.183214 | 0.180867 |
| Pneumonia | 0.216491 | 0.218036 | 0.216239 | 0.190184 | 0.219229 | 0.256572 | 0.259797 | 0.258103 | 0.264871 | 0.255556 |
| Edema | 0.204797 | 0.201649 | 0.202792 | 0.167658 | 0.209570 | 0.276348 | 0.277488 | 0.277759 | 0.299856 | 0.277583 |
| Cardiomegaly | 0.325708 | 0.325988 | 0.324962 | 0.267129 | 0.326489 | 0.422168 | 0.421395 | 0.423214 | 0.453920 | 0.423446 |
| Atelectasis | 0.337646 | 0.335921 | 0.336288 | 0.278409 | 0.341811 | 0.434288 | 0.433796 | 0.434063 | 0.463126 | 0.429027 |
| Lung Opacity | 0.411262 | 0.410365 | 0.409512 | 0.360207 | 0.419411 | 0.503647 | 0.505755 | 0.503052 | 0.519464 | 0.507976 |
| Pleural Effusion | 0.220422 | 0.220700 | 0.219658 | 0.177365 | 0.219435 | 0.357672 | 0.355403 | 0.357556 | 0.380472 | 0.360380 |
| No Finding | 0.260731 | 0.260532 | 0.258513 | 0.185907 | 0.268903 | 0.416401 | 0.420283 | 0.417757 | 0.483276 | 0.428007 |
| Overall BCE | 0.236135 | 0.234986 | 0.234647 | 0.194044 | 0.239685 | 0.311185 | 0.311098 | 0.311292 | 0.331139 | 0.313042 |

## Valid positive/negative BCE and AUROC/AUPRC

| Disease | Experiment | Valid pos BCE | Valid neg BCE | Valid AUROC | Valid AUPRC |
|---|---|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | H0 | 2.667079 | 0.051806 | 0.707859 | 0.097229 |
| Enlarged Cardiomediastinum | H2 | 3.008307 | 0.037094 | 0.717440 | 0.090365 |
| Enlarged Cardiomediastinum | H2B | 2.807019 | 0.044979 | 0.711851 | 0.097006 |
| Enlarged Cardiomediastinum | H2C | 3.120534 | 0.032757 | 0.722156 | 0.089397 |
| Enlarged Cardiomediastinum | H4 | 2.736209 | 0.049013 | 0.715968 | 0.070713 |
| Pneumothorax | H0 | 1.678906 | 0.071985 | 0.855248 | 0.351232 |
| Pneumothorax | H2 | 1.841394 | 0.061487 | 0.856530 | 0.306424 |
| Pneumothorax | H2B | 1.735344 | 0.068975 | 0.854590 | 0.339851 |
| Pneumothorax | H2C | 2.655944 | 0.031098 | 0.836309 | 0.244799 |
| Pneumothorax | H4 | 1.745755 | 0.071367 | 0.854804 | 0.343766 |
| Consolidation | H0 | 2.497378 | 0.047487 | 0.788465 | 0.199799 |
| Consolidation | H2 | 2.562060 | 0.043950 | 0.793732 | 0.197173 |
| Consolidation | H2B | 2.484484 | 0.048048 | 0.790804 | 0.198573 |
| Consolidation | H2C | 2.430144 | 0.056082 | 0.791246 | 0.170530 |
| Consolidation | H4 | 2.453497 | 0.052280 | 0.779159 | 0.189989 |
| Pneumonia | H0 | 2.609858 | 0.054267 | 0.724195 | 0.203233 |
| Pneumonia | H2 | 2.693761 | 0.050556 | 0.718181 | 0.195093 |
| Pneumonia | H2B | 2.639331 | 0.053396 | 0.718487 | 0.196745 |
| Pneumonia | H2C | 2.777722 | 0.048849 | 0.731771 | 0.176412 |
| Pneumonia | H4 | 2.557121 | 0.057697 | 0.722912 | 0.214530 |
| Edema | H0 | 1.053384 | 0.156241 | 0.885361 | 0.478239 |
| Edema | H2 | 1.039401 | 0.159720 | 0.885799 | 0.479321 |
| Edema | H2B | 1.037200 | 0.160372 | 0.886244 | 0.477841 |
| Edema | H2C | 1.205259 | 0.159908 | 0.871406 | 0.462874 |
| Edema | H4 | 1.204389 | 0.134327 | 0.882901 | 0.474027 |
| Cardiomegaly | H0 | 1.094186 | 0.247759 | 0.776882 | 0.430919 |
| Cardiomegaly | H2 | 1.043283 | 0.259996 | 0.782792 | 0.439611 |
| Cardiomegaly | H2B | 1.077514 | 0.253403 | 0.777101 | 0.432624 |
| Cardiomegaly | H2C | 1.140993 | 0.275603 | 0.768536 | 0.415323 |
| Cardiomegaly | H4 | 1.147115 | 0.235631 | 0.777631 | 0.429916 |
| Atelectasis | H0 | 1.183735 | 0.217106 | 0.792594 | 0.462141 |
| Atelectasis | H2 | 1.116494 | 0.235958 | 0.792618 | 0.462046 |
| Atelectasis | H2B | 1.133858 | 0.231269 | 0.793318 | 0.461712 |
| Atelectasis | H2C | 1.119113 | 0.273028 | 0.784791 | 0.450583 |
| Atelectasis | H4 | 1.119325 | 0.228986 | 0.796111 | 0.464587 |
| Lung Opacity | H0 | 1.310825 | 0.235839 | 0.723332 | 0.457379 |
| Lung Opacity | H2 | 1.369211 | 0.219275 | 0.725049 | 0.467758 |
| Lung Opacity | H2B | 1.311650 | 0.234773 | 0.723701 | 0.463479 |
| Lung Opacity | H2C | 1.320575 | 0.253668 | 0.731044 | 0.465007 |
| Lung Opacity | H4 | 1.376992 | 0.219652 | 0.721043 | 0.453152 |
| Pleural Effusion | H0 | 0.685289 | 0.215468 | 0.908774 | 0.776823 |
| Pleural Effusion | H2 | 0.605639 | 0.246786 | 0.909574 | 0.784624 |
| Pleural Effusion | H2B | 0.655072 | 0.228416 | 0.909147 | 0.776433 |
| Pleural Effusion | H2C | 0.642598 | 0.266694 | 0.905943 | 0.778655 |
| Pleural Effusion | H4 | 0.686151 | 0.218977 | 0.911150 | 0.799355 |
| No Finding | H0 | 0.688984 | 0.265568 | 0.871979 | 0.827054 |
| No Finding | H2 | 0.728522 | 0.249720 | 0.868666 | 0.827164 |
| No Finding | H2B | 0.712482 | 0.254673 | 0.871603 | 0.828980 |
| No Finding | H2C | 0.924388 | 0.239188 | 0.861805 | 0.823962 |
| No Finding | H4 | 0.681653 | 0.287653 | 0.869015 | 0.818398 |

## Text analysis

- **H2B**: Train BCE 下降但 Valid BCE 上升的疾病：Pneumonia、Edema、Cardiomegaly、No Finding；Valid BCE 下降且 AUROC 提高的疾病：Enlarged Cardiomediastinum、Consolidation、Atelectasis、Lung Opacity、Pleural Effusion。
- **H2C**: Train BCE 下降但 Valid BCE 上升的疾病：Pneumothorax、Consolidation、Pneumonia、Edema、Cardiomegaly、Atelectasis、Lung Opacity、Pleural Effusion、No Finding；Valid BCE 下降且 AUROC 提高的疾病：Enlarged Cardiomediastinum。
- **H4**: Train BCE 下降但 Valid BCE 上升的疾病：Consolidation、Pleural Effusion；Valid BCE 下降且 AUROC 提高的疾病：Enlarged Cardiomediastinum、Atelectasis。
- AUROC 提高但 AUPRC 下降的组合：H2B:Enlarged Cardiomediastinum、H2B:Consolidation、H2B:Edema、H2B:Atelectasis、H2B:Pleural Effusion、H2C:Enlarged Cardiomediastinum、H2C:Consolidation、H2C:Pneumonia、H4:Enlarged Cardiomediastinum、H4:Cardiomegaly。这提示部分概率尺度/校准变化不能解释为排序能力提升。
- 单图与多图子群变化并不总是一致；见 `grouped_performance_comparison.csv`，不能把 Study 级 Macro-AUROC 当作子群 AUROC 的加权平均。

## Every-epoch Validation curves

- 逐 epoch 十类 Validation AUROC/AUPRC/BCE：新实验的 `per_epoch_class_metrics.csv`；H0/H2 由历史 `val_epoch_*.csv` 重建，见 `epoch_class_metrics/`。
- 总体与逐疾病曲线：`overall_loss_curves.png`、`per_disease_loss_plots/`、`per_disease_auroc_plots/`。

## Assumptions and limits

- 训练批次 BCE 与 Validation 确定性 BCE 不可直接比较；所有表格已标注口径。
- H0 为复用 R5；H2 为历史实验；新实验均从原始 DINOv3 权重独立训练，未叠加。
- 单 seed；无 Test 评估。

