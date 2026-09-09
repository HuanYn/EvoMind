# 2026-09-09 执行范围修订

用户明确取消 PPO、Agent-GRPO、人工盲评；RLAIF 的 2,000 条提议撤回，保留完整 19,502 条。所有原始数据和已有产物保留，不删除旧实验。

## 权重关系

预训练 → full_sft → 六条独立后训练分支：DPO、LoRA、GRPO、CISPO、Agent-CISPO、蒸馏。不是 DPO → GRPO → CISPO 的连续训练。LoRA 推理使用 SFT + adapter；蒸馏学生从 SFT 初始化，另用已核验的官方公开 MoE 教师。单张 GPU 只决定排队运行顺序。

## 评测

- 固定 MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba` 的 `eval_llm.py` 八条自动问答：保存原始回答、思考开关、生成日志；seed42替代官方随机seed，记录这项可复现性适配。温度0.85、top_p0.95、默认top_k50、生成上限8192。
- 官方 `scripts/eval_toolcall.py` 八例：本地工具后端，seed42，现有Windows安全边界保留。这不是20题数学benchmark，也不是通用Agent成功率。
- 官方README列出的7项：C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social-IQA。使用本机 lm-evaluation-harness 0.4.13 的原生任务；完整split、task默认few-shot、apply_chat_template、关闭思考。原生HF兼容模型可直接交给HFLM，LoRA仍绑定同一SFT基座。FP16、batch1为本机显存适配，版本/源码/权重/模板哈希和逐题输出写入结果；不保证和官方已公布数值完全一致。
- 旧自写中文评分器不再排队；自编20题、多解码网格、自编500图评测不再作为未来验收项。保留历史代码和结果，但不与新协议混表。
- 固定 MiniMind-V commit `740d467ece78a0b7d2d976fcb424472095d4a688` 的六张示例图及描述prompt，生成上限512、温度0.7、top_p0.85。多视图是evomind扩展，不冒充原始单图实现。C组对新示例图使用冻结编码器在线推理，因此不能用这次推理速度证明缓存加速。

人工盲评是“取消”，不是“通过”。机器评测缺项、失败、样本不足或哈希变化仍阻止验收。训练完成也不意味着模型达到求职展示质量；图文示例没有独立正确性标注，不能声称有质量胜率。

## 曲线

预训练/SFT继续保留 `artifacts/runs/text_official_mini_20260908/metrics/*.jsonl` 和 `curves/*.png`。各后训练分支保留独立目录的 `console.log`、`observed.metrics.jsonl`、`curves.png` 与原生运行回执。视觉训练日志/曲线和聚合图继续保留，不因取消额外评测而停止记录。空白或失败的曲线不能冒充已完成结果。
