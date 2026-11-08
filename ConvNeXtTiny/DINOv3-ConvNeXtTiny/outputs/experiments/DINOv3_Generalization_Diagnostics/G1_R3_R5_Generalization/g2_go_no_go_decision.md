# G2 go / no-go decision (after G0 + G1)

- G0 gate passed: **True**.
- Train−Valid Macro-AUROC gap: R3 = +0.099156, R5 = +0.099627.
- Stage4 parameter relative drift: R3 = 0.007984, R5 = 0.008006.
- Stage4 GAP raw feature cosine vs pretrained: R3 = -0.0126, R5 = 0.1340; classifier-input (final) feature cosine: R3 = 0.0704, R5 = 0.2087.

## Decision

**RECOMMEND G2**

建议在用户确认后启动 G2（1e-5 vs 3e-6 配对种子实验）。依据：Train−Valid Macro-AUROC 差距 R3=+0.0992/R5=+0.0996；Stage4 参数相对漂移 R3=0.0080/R5=0.0080；分类器输入(final)特征余弦 R3=0.0704/R5=0.2087（原始 Stage4 GAP 余弦 R3=-0.0126/R5=0.1340）。这些是可检验“微调强度影响泛化”的合理依据，但不构成因果证明。

## Conditions for G2 (if approved)

- Training matrix is fixed in advance: R5 Global-only, Stage3/4 LR 1e-5 vs 3e-6, seeds 42/2026/3407, head LR 2e-4; no extra LR grid.
- Seed 42 baseline (`G2-A1`) may reuse R5 only if the protocol is identical; otherwise re-run it. All other runs must be freshly trained.
- G2 has NOT been started and must not start without explicit user confirmation.

## What would change this decision

- If a distribution/representativeness investigation shows the Validation set overestimates Test performance, lowering LR is unlikely to fix the Test gap by itself.
- If Train−Valid gaps were small, the priority would be Test distribution analysis rather than G2.

