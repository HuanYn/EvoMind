# 文本基座选择与视觉交接

## 选择结论

2026-09-12，按项目预先配置的工程筛选规则，选择最终 **CISPO** 权重作为 Dense 单图实验的初始化。GRPO 保留为同一 DPO 父权重出发的对照，不作为串行训练的下一环。

```text
SFT → DPO → CISPO → Dense 单图
         └→ GRPO（对照）
```

选中权重 SHA256：`171d23ad98338540617532ca101c47045210203c798547e781141dcb77d0f1ed`。

## 为什么选它

不是按训练 reward 排名。规则要求七项原生 `acc,none` 各自下降不超过 1 个百分点，EOS 不下降、repeat-4 不增加，并至少一项改善达到 0.001。

| 检查项 | CISPO 相比 DPO | 判断 |
|---|---:|---|
| 七项基准最大下降 | CMMLU 约 0.095 个百分点 | 在既定容差内 |
| 八条 thinking-off 回答 EOS | 8/8 → 8/8 | 不下降 |
| thinking-off repeat-4 | 0.113065 → 0.089359 | 下降约 0.023706 |
| 全量训练 | 19,502 条问题、1 epoch、9,751 次更新 | 非 smoke 权重 |

同父权重的 GRPO 也通过 DPO 相对筛选。继续使用 CISPO 是既定主线的工程选择，不是 CISPO 显著优于 GRPO 的结论。单随机种子、八条开放生成样例、不同机器的 harness 源码哈希差异均保留。ToolCall 完成不代表回答正确；已观察到错误结果。

## 如何验收

`scripts/evomind_accept_text.py` 导入既有结果，不重新训练或重新评测：

1. 校验 SFT、DPO、CISPO、GRPO 检查点及训练完成回执，拒绝探针权重和错误父权重。
2. 校验 28 组原始 `results.json`、完整 split 覆盖、摘要与原始成绩一致性，保留原始评测合同。
3. 校验 thinking 开/关的原始记录与工具诊断，重新计算筛选值。
4. 只允许数据归档路径搬迁，不忽略数据、模型、tokenizer 或评测配置变化。评测时的旧加载器单独保留，避免把后来的网页加载兼容修复混入历史合同。
5. 保留旧产品状态，写入带哈希约束的 `product_acceptance.json`；再次核验后才开启视觉训练。

运行前必须准备含原始结果和权重的导入清单。下面只检查，不开启训练：

```bash
python scripts/evomind_accept_text.py --spec artifacts/handoff_20260912/import_spec_v3.json --check-only
```

去掉 `--check-only` 会记录验收、更新当前产品状态并开放视觉门槛，但本身不启动 GPU 训练。权重、原始数据、机器路径和完整回执不上传 GitHub；公开结果见 [文本证据快照](results/EVIDENCE_20260911.md)。

## Dense 单图实验合同

- SigLIP2 视觉塔保持冻结，输入一张完整图片，256×256、64 个视觉 token。
- `freeze_llm=1`：训练 Projector 以及语言模型首尾两个 Decoder Block；其余语言模型参数冻结。这不是全参数微调，也不是只训练 Projector。
- 2,544,979 条 image-group-disjoint 训练样本、2 epoch、序列长度 768、BF16、学习率 `5e-6`。
- 优先 microbatch 4、累积 1；只有实测 OOM 后才采用 microbatch 1、累积 4，保持有效 batch 4。
- 先做独立两步显存/反向传播检查，再从未更新过的已验收基座启动正式训练。
- 记录控制台、指标 JSONL、曲线、训练合同和检查点。原生恢复包含模型/优化器/scaler/epoch/step，但不承诺数据增强 RNG 的逐位重放。

该实验是图像基线。视频训练、MoE 对照与视觉理解质量仍需各自的训练和评测证据，不能由文本验收代替。
