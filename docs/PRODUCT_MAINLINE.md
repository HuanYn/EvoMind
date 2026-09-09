# evomind 产品主线

## 2026-09-09：evomind 产品主线定版（取代六分支必跑）

目标是形成可用的视频理解项目，不再以跑齐算法清单作为完成标准。

```text
文本预训练 → SFT → DPO → 验收选 SFT/DPO → CISPO → 验收选择
                                                    ├ 图文 → 视频【交付主线】
                                                    └ Agent-CISPO【延后工具扩展】
```

- DPO 有改善才作为 CISPO 初始化；否则回到已验收 SFT。CISPO 也必须验收，不因 reward 上升就自动晋级。
- 这是 evomind 自定义阶段衔接，不是 MiniMind 默认独立分支的原样复现。
- Agent 保留，但不阻塞视觉。视频 Agent 必须有实际视频工具/任务，不能拿数学工具成绩冒充。
- LoRA、独立 GRPO 和完整蒸馏暂缓；PPO、Agent-GRPO 维持取消。已生成产物不删。
- 保持全量 19,502 条 RLAIF、既定 epoch/奖励配置、训练日志/曲线/权重，当前 SFT 不重启。
- 继续官方八问（思考开关）、八个工具例子和七项中英文 harness 客观评测，不恢复人工盲评。
- 自动验收仅为工程筛查：SFT 非思考八问 EOS≥50%、平均 token repeat-4≤50%；候选每项客观准确率最多下降1个百分点，EOS不降、复读不升，并至少一项改善0.1个百分点才采用。失败/缺测不算通过；这些阈值不是官方标准、统计显著性或语义正确性证明。
- 现有图像流水线不等于视频模型；视频数据合同、时序输入与视频训练/验收仍待实现，不能在图像阶段结束就宣称项目完成。

执行合同：`E:/project/Learning/evomind/configs/product_pipeline.json`。
自动入口：`scripts/evomind_continue.py` → `scripts/evomind_product.py`。
旧六分支合同保留作历史资料，不再自动调度。
