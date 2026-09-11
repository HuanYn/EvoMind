# GRPO/CISPO 多轮 rollout-reuse 机制短消融

日期：2026-09-11

范围：**机制验证，不是产品模型选择，也不是通用能力测评。**

## 目的

本实验复现已有的 rollout-reuse 与剪裁机制，不将其作为本项目提出的新方法。正式文本分支每个 rollout 只更新一次（`updates_per_rollout=1`）。在这种设置中，当前策略与生成时冻结的旧策略几乎相同，重要性比率

$$
r_t=\exp(\log\pi_\theta(y_t)-\log\pi_{\mathrm{old}}(y_t))
$$

接近 1，GRPO 的窄裁剪和 CISPO 的宽上限基本不会介入。

本实验将每组已生成、已奖励的 rollout 固定复用四次（`K=4`）。第 1 次更新后参数改变，第 2 至第 4 次重新计算同一输出的 `logp`，从而观察两种算法在 ratio 偏离 1 后的实际行为。

## 受控设置

| 项目 | 设置 |
|---|---|
| 初始权重 | 同一 DPO：`dpo_768.pth`，SHA256 `ccba444524c69d19a936646f296b89c13c3ee5bdc3f6cd30a037925a5b15ca2a` |
| 数据源 | 同一 `dataset/rlaif.jsonl`（19,502 条）；每分支实际使用 30 组 prompt，并非遍历整个数据集 |
| 两条分支 | GRPO / CISPO；各 120 optimizer updates，即 30 rollout groups × K=4 |
| 生成 | G=6，prompt 768，最大生成 1024 |
| 优化 | batch=1、accumulation=1、LR=1e-6、beta=0.1、seed=42 |
| 阈值 | GRPO `epsilon=0.2`；CISPO `epsilon_high=5.0` |
| 运行位置 | GPU 2：CISPO；GPU 3：GRPO；两张 RTX 3090 |

`1e-6` 是为了在短实验内让 `r_t` 有机会跨过 GRPO 的 1.2/0.8 阈值而采用的机制敏感性设置；它不是正式主线的 `3e-7` 训练配置。

## 指标定义

对于正 advantage，GRPO 在 `r_t>1.2` 后的 policy 项成为常数；对负 advantage，`r_t<0.8` 后成为常数。这些位置的 GRPO policy gradient 被抑制（KL 项仍存在）：

$$
\mathrm{GRPO\ suppressed}=
\mathbb{1}[A>0,r>1.2]\ \lor\ \mathbb{1}[A<0,r<0.8].
$$

CISPO 只在 `r_t>5` 时停止提高重要性权重：

$$
\mathrm{CISPO\ capped}=\mathbb{1}[r>5].
$$

`native_intervention_rate` 取各自损失真正使用的那个统计：GRPO 取 suppressed rate；CISPO 取 capped rate。

## 已得到的结果

| 指标（120 次更新均值） | CISPO | GRPO |
|---|---:|---:|
| ratio p95 | 1.0370 | 1.0371 |
| ratio max（逐 update max 的均值） | 1.1700 | 1.1556 |
| GRPO suppressed rate | 0.5211%（反事实） | 0.5507%（原生） |
| CISPO capped rate | 0.0000% | 0.0000%（反事实） |
| native intervention rate | 0.0000% | 0.5507% |
| reference KL penalty | 0.00298 | 0.00292 |

按同一 rollout 的复用轮次看，GRPO 原生抑制率从第 1 次的 0 上升到第 4 次的 **0.9801%**；CISPO 在第 4 次的反事实 GRPO 抑制率为 **0.8200%**，但其自身的 5.0 cap 仍未触发。

## 结论与边界

1. 多轮 rollout-reuse 后，ratio 随策略更新偏离 1，GRPO 的裁剪机制确实被触发；因此该设置可用于展示两种优化目标的机制差异。第 1 次更新的理论比率为 1，实际计算可能有浮点误差。
2. CISPO 通过停止梯度的重要性权重训练有效 token 的 log-prob；即使达到 cap，也不等于切断该 log-prob 的梯度。GRPO 则对一小部分 token 的 policy 项抑制梯度，KL 项仍存在。这验证了不同的梯度处理方式，不等于证明最终质量或统计意义上的样本效率收益。
3. 平均 KL 同数量级，短程未见明显不稳定。但 **120 updates 不能证明 CISPO 的最终回答质量或通用 benchmark 一定更好**。
4. 两个分支在独立 GPU 上采样，起点和配置相同，但不应宣称逐 token 的比特级相同。若要报告显著性或最终质量优势，需要多随机种子、固定 rollout 缓存或更长的预注册训练对照。

## 产物

本次发布包含[逐步数值](../results/ratio_reuse_k4_20260911.jsonl)、[汇总](../results/ratio_reuse_k4_20260911.json)和[曲线](../assets/ratio_reuse_k4_20260911.png)。无需 GPU 的复画步骤见[证据说明](../results/EVIDENCE_20260911.md)。

完整原始运行产物保留在实验机器的以下目录，不随普通 Git 提交上传：

- `artifacts/diagnostics/ratio_reuse_k4_lr1e6_20260911/cispo/`
- `artifacts/diagnostics/ratio_reuse_k4_lr1e6_20260911/grpo/`
- `artifacts/diagnostics/ratio_reuse_k4_lr1e6_20260911/summary.json`
- `artifacts/diagnostics/ratio_reuse_k4_lr1e6_20260911/curves.png`

复现配置：`configs/ratio_reuse_ablation_k4_20260911.json`。
