# 2026-09-09 评测恢复记录

## 已观察到的问题

1. C-Eval 在 Hugging Face 数据访问阶段报 `ConnectionResetError(10054)`，不是模型推理或权重错误。
2. CMMLU 的 `datasets.load_dataset` 未启用自定义加载脚本信任，抛出 `trust_remote_code=True` 要求。当前 datasets 为 3.6.0；不升级或改动基础环境。
3. ARC-Easy 已完成 loglikelihood 推理，但 JSON 序列化不支持 `torch.dtype`。没有成功保存的 results/summary，不能记为有效成绩。
4. 15:18 停止了经 PID/父进程/命令行校验的本项目 continuation（48132）及其受管理评测子进程。当时 PIQA 尚未完成；不标为程序自身崩溃。SFT 已于 14:57 完成，没有重训或修改其权重。
5. 首次恢复于 15:25 遇到监督进程 GBK 控制台编码错误（特殊字符无法编码）；追加 `scripts/resume_evomind.ps1` 为整个子进程树设置 UTF-8，15:26 再次按断点恢复。首次恢复日志仍保留；不将它记作模型推理失败。

## 修复方式与结果边界

- 新增 `scripts/evomind_harness_data.py`：读取已经下载且校验 SHA256 的官方原始 CSV 压缩包。按上游 pandas 读取方式及 datasets 字段类型构造数据，包含完整评测 split 和 dev。不是调用旧本地 MCQ 的提示词/计分函数，也不读取旧标准化 records。
- C-Eval 固定源 revision `3923b519fd180e689d0961bf3a032ece929742f3`；CMMLU 固定源 revision `efcc940752ea4a1ea94d2727f11f83858d64fc8e`。新协议记录加载适配代码及源哈希。此版本与随 HF main 更新的数据来源不混为逐位相同。
- 保留原生 harness 任务、描述、few-shot 默认值、模板、候选评分、样本数量与随机种子。不联网执行数据集 Python，不全局打开任意远程代码信任。
- JSON 显式支持 dtype/device/path 元数据以及 numpy 数值，继续拒绝未知类型和非有限数值，不能把错误静默字符串化。
- 新的 harness 产物使用 `_v2` 目录，旧日志/contract 保留。已完成 thinking 开/关结果继续校验复用。
- 修复回归 7 项、产品主线 12 项、既有范围 5 项测试通过（24 项）。另运行真实原生任务 CPU 数据预检，结果见 `artifacts/benchmarks/harness_local_preflight.json`；预检不等于模型评测。
- 完整成绩仍以恢复后的 results.json / summary.json 与全覆盖校验为准，不从进度条或控制台猜测分数。
- 规范后台恢复入口为 `scripts/resume_evomind.ps1`：启动前拒绝重复进程，保存恢复前状态快照，UTF-8 隐藏启动，日志在 E 盘独立目录；不修改全局环境。最新启动目录为 `artifacts/runs/recovery_20260909_152635_617`。

## 兼容性查证（web-debug-search）

| 匹配 | 来源 | 结论与限制 |
|---|---|---|
| [COMPATIBILITY] | [harness 官方配置实现](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/config/evaluate_config.py) | CLI 设置包含显式信任配置；直接调用 Python API 不应假设已经应用 CLI 设置。实际修复使用本地原始数据适配。 |
| [CONTEXTUAL][DEBUGGING] | [harness issue #1135](https://github.com/EleutherAI/lm-evaluation-harness/issues/1135)（2023-12-15 创建，Closed） | 维护者解释 datasets 引入远程脚本信任要求；是历史兼容性线索，不是本次修复成功证据，也没有确认与本机所有版本完全相同。 |

Evidence boundary: 这些 GitHub/web 结果仅用于调试和兼容性发现，不是论文引文证据，不加入研究参考文献，也不单独支撑模型效果结论。修复效果以本机预检与正式运行验证。
