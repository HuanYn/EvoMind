# MiniMind-3 文本训练配方审计

审计日期：2026-09-08。上游：`jingyaogong/minimind`，固定提交 `6fc918beb68a0d8c40452338df6319fe168014ba`。本报告只依据该提交的 README、训练脚本、数据集类、模型与 rollout 实现，不依据旧版本教程推断训练顺序。DPO/PPO/GRPO/Agent/LoRA/蒸馏脚本及 README 在本次审计时与固定提交无工作区差异；本地 `trainer_utils.py`、pretrain/SFT 已有 evomind 的严格加载和保存安全修复，不能把这些本地修复称为上游默认。

本次没有启动 GPU、下载数据、改动训练代码或当前预训练进程。此前监督器的 Windows Job Object 修复已通过 17 项 CPU 测试，包括真实 venv 启动器子孙进程的中断清理、owner 突然退出时自动清树，以及嵌套 continuation 的显式恢复传播。

## 1. 哪些是主线，哪些是分支

README 的主线推荐是 **`pretrain_t2t` → `sft_t2t` → `rlaif/agent_rl`**。快速 Zero 路线使用 `pretrain_t2t_mini` 和 `sft_t2t_mini`。README 将知识蒸馏、LoRA 放在“其它训练（可选）”，将 DPO/PPO/GRPO/CISPO/AgentRL 放在“强化学习（可选）”；这里“可选”表示上游没有要求每个发布模型都经历所有分支，并不表示本项目不能单独复现这些分支。[主线建议](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L542)、[可选训练](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L731)、[可选强化学习](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L887)。

主要展示的后训练比较是 **SFT、GRPO、Agent-CISPO**。所有被审计的后训练脚本都默认直接从 `full_sft` 初始化；没有代码证据要求 `DPO → PPO → GRPO → CISPO → Agent → LoRA → 蒸馏` 顺序串联，也没有公开训练清单证明发布权重经历了该串行路线。README 还明确说明部分分支权重不持续发布。[对比对象](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L1315)、[发布范围说明](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L1286)。

因此，若复现“所有文本分支”，应保留同一份 `full_sft_768.pth` 作为各分支起点，并给分支独立命名、记录继承关系、单独评价。若主动将某分支输出作为下一分支输入，那是明确配置的新组合，不能称为代码默认继承链。

## 2. 可执行默认值与 README 配方

表中 B 为每个数据加载批次的记录/提示数量，A 为梯度累积参数。除白盒蒸馏教师外，默认语言模型都是 Dense、hidden=768、8 层、6400 词表。各脚本默认 BF16、`grad_clip=1`、不启用 compile、`from_resume=0`；下列后训练分支的 `num_workers` 都为 8。训练脚本按上游要求从 `trainer/` 执行。

| 分支 / 脚本 | 默认继承与额外模型 | 默认数据 | epochs / B / A | 学习率与长度 | README 的配方/定位 |
|---|---|---|---|---|---|
| 预训练 `train_pretrain.py` | `from_weight=none`，随机初始化 | `pretrain_t2t_mini.jsonl` | 2 / 32 / 8 | LR `5e-4`；seq 340 | 必需基础阶段；完整主线推荐 `pretrain_t2t`。README 对 mini 推荐约 768 tokens，与脚本 340 不一致；耗时估算按 1 epoch，与脚本 2 不同 |
| SFT `train_full_sft.py` | `pretrain_768.pth` | `sft_t2t_mini.jsonl` | 2 / 16 / 1 | LR `1e-5`；seq 768 | 必需基础阶段；完整主线推荐 `sft_t2t`。Tool Call 和教师生成的 reasoning 数据已混入 |
| DPO `train_dpo.py` | policy 与冻结 reference 都读 `full_sft_768.pth`；没有独立 RM/critic | `dpo.jsonl`，`chosen`/`rejected` 对 | 1 / 4 / 1 | LR `4e-8`；seq 1024；beta `0.15` | `python train_dpo.py`，未额外覆盖 epochs；可选静态偏好对齐分支 |
| PPO `train_ppo.py` | actor、冻结 reference、critic 主干均来自 `full_sft_768.pth`；critic 新建 value head；额外 InternLM2-1.8B Reward | `rlaif.jsonl` | 1 / 2 / 1 | actor LR `3e-7`、critic LR `5e-7`；prompt 768、生成最多 1024 | `python train_ppo.py`，未额外覆盖 epochs；独立 RLAIF 分支 |
| GRPO `train_grpo.py --loss_type grpo` | policy 与冻结 reference 都读 `full_sft_768.pth`；额外 RM；没有 critic | `rlaif.jsonl` | 1 / 2 / 1 | LR `3e-7`；prompt 768、生成最多 1024；每 prompt 6 个回答 | README 的 GRPO 段落给出裸 `python train_grpo.py`；但固定代码的默认 loss 实际是 CISPO。复现 GRPO 必须显式指定 `--loss_type grpo` |
| CISPO `train_grpo.py` | 与 GRPO 同一模型/数据/采样管线 | `rlaif.jsonl` | 1 / 2 / 1 | 同 GRPO；默认 `loss_type=cispo` | README 将 CISPO 描述为 GRPO 的 loss 变体，不是必须接在 GRPO 后的独立阶段。默认仍保存 `grpo_768.pth`，文件名不能证明算法是 GRPO |
| AgentRL `train_agent.py` | policy 与冻结 reference 都读 `full_sft_768.pth`；额外 RM 默认无条件加载；没有 critic | `agent_rl.jsonl`，含工具 schema 与 `gt` | 1 / 2 / 1 | LR `3e-7`；配置 prompt 1024、每轮生成最多 768、训练最终总长最多 2500；每 prompt 4 条轨迹 | 裸 `python train_agent.py`；默认 `loss_type=cispo`。README 的 SGLang 示例改为 `agent_rl_math.jsonl`；这属于另一个数据/rollout 配置 |
| LoRA `train_lora.py` | 冻结 `full_sft_768.pth` 主体；只训练新增 LoRA；无教师/reference/RM | `lora_medical.jsonl` | 10 / 32 / 1 | LR `1e-4`；seq 340；rank 固定调用默认 16 | `python train_lora.py`，未额外覆盖 epochs；医疗/身份等垂域适配，不是通用发布模型的必经阶段 |
| 白盒蒸馏 `train_distillation.py` | 学生 Dense `full_sft_768.pth`；教师 MoE `full_sft_768_moe.pth`，两者均 768×8；教师冻结 | `sft_t2t_mini.jsonl` | 6 / 32 / 1 | LR `5e-6`；seq 340；alpha `0.5`、T `1.5` | `python train_distillation.py`，未额外覆盖 epochs；README 明确称为白盒蒸馏参考实现。只有 Dense SFT 权重不能满足默认教师依赖 |
| Tool Call / adaptive thinking | 同一 SFT/RLAIF/Agent 模型及 chat template | 已混入 SFT 的 tools、tool_calls、reasoning_content；RLAIF prompt 开关采样 | 没有独立 epoch 配方 | PPO/GRPO `thinking_ratio=0.9`；Agent `0.1`；推理 `open_thinking` | 2026-03 起移除独立 `train_reason.py`；无须另排一个“reason 模型训练”或“toolcall 专用 SFT”阶段 |

默认值位置：[DPO 133 起](../trainer/train_dpo.py)、[PPO 311 起](../trainer/train_ppo.py)、[GRPO/CISPO 208 起](../trainer/train_grpo.py)、[Agent 374 起](../trainer/train_agent.py)、[LoRA 80 起](../trainer/train_lora.py)、[蒸馏 149 起](../trainer/train_distillation.py)。基础阶段默认值以固定提交的对应 parser 为准；当前本地两个基础训练文件已经包含安全修复。

保存/日志频率（单位是外层数据批次 step，不自动等于 optimizer step）：DPO 和蒸馏 `log=100/save=100`；LoRA `log=10/save=1000`；PPO、GRPO/CISPO、Agent `log=1/save=10`。输出前缀分别是 `dpo`、`ppo_actor`、`grpo`、`agent`、`lora_medical`、`full_dist`。

## 3. 权重和算法继承的精确含义

**DPO。** 两个模型由同一个 `args.from_weight` 初始化；reference 设为 eval 且关闭梯度。每条偏好记录拆为 chosen/rejected，默认 B4 会拼成 8 条序列进行 policy/reference 前向；数据集的模型输入实际是截断长度减 1。实现对 assistant mask 内的 token log-prob 求和，再优化 `-logsigmoid(beta * ((logπchosen-logπrejected) - (logπrefchosen-logπrefrejected)))`，加 MoE aux loss。README 提到 reference 可预缓存，但实际默认代码每批重新执行 reference 前向，没有缓存路径。来源：`train_dpo.py:34,66,75,191`、`dataset/lm_dataset.py:154`。

**PPO。** actor 和 reference 用 `init_model`；critic 用同一基模 state dict 做 `strict=False` 加载，因此新的 `value_head` 随机初始化，不继承任何已训练 reward head。RM 是独立的 InternLM 模型，不是 critic。每个 prompt 生成 1 个回答，温度 0.8；每批 rollout 重复更新 2 轮，更新 minibatch 默认 2。GAE `gamma=1`、`lam=0.95`；policy clip `0.2`、value clip `0.2`、value loss 系数 `0.5`、reference KL 系数 `0.02`、approx-KL early stop 阈值 `0.25`。外部轨迹 reward 加在最后有效 response token，再计算 GAE；优势做有效 token 标准化。critic 的 forward 使用自定义 value head；代码还在 `self.model` 返回后再次调用 `model.norm`，复现时不应无记录替换成其他 critic 架构。来源：`train_ppo.py:36,89,133,200,330,380`。

**GRPO / CISPO。** 默认 B2×G6=12 条候选序列；rollout 温度 0.8。组内优势为 `(reward-group_mean)/(group_std+1e-4)`，std 使用 `unbiased=False`。reference KL 是 `exp(ref_logp-policy_logp) - (ref_logp-policy_logp) - 1`，系数 `beta=0.1`。GRPO clip 为 `[1-0.2,1+0.2]`；CISPO 将 ratio 上限裁为 `epsilon_high=5.0` 并 detach，乘 advantage 和 policy log-prob。按每条回答有效 token 平均，再对回答平均。`train_grpo.py` 默认保存名 `grpo` 对两种 loss 都适用，独立比较时须改 `--save_weight` 以免覆盖。来源：`train_grpo.py:78,108,127,209,226`。

**AgentRL。** 默认 B2×G4=8 条轨迹。rollout 循环对每个 prompt/每条轨迹逐一生成；每条最多 3 个 assistant 轮次，`max_turns=3` 在调用处写死，不是 CLI 选项。每轮温度 0.8、最多 768 新 tokens；工具结果字符串截到 2048 字符。工具 observation 对应位置不参与 policy loss。模型训练前将打包后的 `prompt+responses+observations` 从左截断到 2500 tokens。`max_seq_len=1024` 被写入 config，但 Agent 数据集只是存储 `max_length`，rollout tokenization 并未据此截断初始 prompt，因此不能把 1024 当作真实 rollout 峰值上下文上限。CISPO/GRPO loss、组内标准化与 KL 系数和上一分支相同，默认 CISPO。来源：`train_agent.py:98,143,159,242,251,259,312,389`、`dataset/lm_dataset.py:227`。

**LoRA。** `apply_lora(model)` 默认 rank 16，只挂到 `nn.Linear` 且输入维度等于输出维度的层；在默认 Dense 架构中主要是 attention 的 `q_proj/o_proj`，不是任意“全线性层 LoRA”。增量为 `B(A(x))`，没有另一个 CLI alpha/r 缩放或 LoRA dropout；A 高斯初始化、B 零初始化。非 LoRA 参数全部冻结。导出 `lora_medical_768.pth` 仅含 LoRA 参数；恢复文件另含完整模型/优化器状态。代码遇 `use_compile=1` 会明确自动关闭 compile。来源：`model/model_lora.py:5,21,45`、`train_lora.py:128,139,165`。

**白盒蒸馏。** 学生 Dense 默认约 64M，教师 MoE 约 198M-A64M；不是让同一个 Dense checkpoint 教自己。总损失为 `0.5 * CE + 0.5 * T² KL(teacher/T || student/T)`，T=1.5，只在 assistant 的有效 label token 上蒸馏；MoE 学生的 aux loss 进入 CE 支路。教师前向处于 no-grad/eval，但在学生 autocast 区块之外执行，默认教师参数由 `init_model` 构造为 FP32。教师和学生共享同一 tokenizer/token id 语义，代码只截取 teacher logits 的前 student-vocab-size 维，不提供异构 tokenizer 对齐。来源：`train_distillation.py:25,45,56,88,92,163,207`。

**学习率调度。** DPO、LoRA、蒸馏沿用 `get_lr`：`lr*(0.1+0.45*(1+cos(pi*current_microstep/total_microsteps)))`，无 warmup。PPO、GRPO/CISPO、Agent 用 `CosineAnnealingLR`，在 optimizer step 时推进，最低设为初始 LR/10。GRPO/Agent 的 T_max 按 `ceil(loader_steps/accumulation_steps)*epochs`；PPO 还计入每 rollout 的更新轮次和 minibatch 数，early stop 可能使实际更新次数少于名义总数。不能把所有脚本的外层 step 直接画成同一种 optimizer-step 横轴。

## 4. 奖励模型与规则奖励

PPO、GRPO/CISPO、Agent 的默认 RM 路径都为 `../../internlm2-1_8b-reward`（相对 `trainer/`，即项目同级目录）。`LMForRewardModel` 用 `AutoTokenizer/AutoModel.from_pretrained(..., trust_remote_code=True)`，FP16，eval，单独调用它的 `get_score`，把每个 RM 分数裁到 `[-3,3]`。所有这些默认入口都将 RM 放到同一个 `args.device`，没有默认 CPU reward offload 或禁用 RM 的 CLI。Agent 的奖励函数签名允许 `None`，不代表其主程序支持省略 RM；主程序实际无条件创建它。来源：固定提交 [`trainer_utils.py`](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/trainer_utils.py)、`train_ppo.py:391`、`train_grpo.py:279`、`train_agent.py:440`。

| 分支 | 规则项与 RM 项 | 最后是否再裁总分 |
|---|---|---|
| PPO、GRPO/CISPO | 完整 response 长度 20–800 **字符**得 +0.5，否则 −0.5；存在 `</think>` 时，thinking 长度 20–300 字符得 +1，否则 −0.5；闭合标签恰好 1 个得 +0.25，否则 −0.25；去掉 thinking 后的 answer 减 3-gram 重复惩罚（上限 0.5）；加 RM 分数 | 不再整体裁分；RM 自身先裁 `[-3,3]`，总 reward 可超过 3 |
| Agent：没有解析出工具调用 | 先扣未配对 `<tool_call>` 标签每处 0.5；response 长度合法区间改为 5–800 字符，得 +0.5/−0.5；thinking 两项同上；加 RM；扣重复惩罚 | 总分裁 `[-3,3]` |
| Agent：有工具调用 | 同样扣 tool tag 不闭合；合法调用数由样本允许工具名 + 必填参数检查决定。`tool_gap=abs(valid_call_count-len(gt))+invalid_call_count`，为 0 加 0.5，否则减 `0.5*tool_gap`；最终文本命中 GT 按 `2.5*命中数/GT数` 加分；未完成多轮调用扣 0.5；扣重复惩罚 | 总分裁 `[-3,3]`；此分支不加 RM 分 |

来源：`train_ppo.py:29,51`、`train_grpo.py:31,37`、`train_agent.py:183,188`。

Agent 的 GT 校验是子串匹配或数值匹配（绝对误差 <1e-6），不是数学证明判定；工具调用合法性奖励检查名称和参数，不直接证明工具执行结果正确。默认 6 类工具是 `calculate_math`、`unit_converter`、`get_current_weather`、`get_current_time`、`get_exchange_rate`、`translate_text`。天气/时间/汇率/翻译是脚本内固定 mock 数据，不访问真实外部服务。不能把这里的 reward 当成人工标注准确率、真实天气/汇率工具能力或广义 Agent 成功率。

## 5. 自适应 thinking 与工具调用

- 推理 `open_thinking=1`：chat template 预填 `<think>\n`；`0`：预填空 `<think>\n\n</think>\n\n`。这是同一模型的提示格式开关，不会凭空增加权重能力。
- SFT 的模板接受 `reasoning_content` 或内容中的 think 标签；`post_processing_chat` 会以 80% 概率移除空 thinking 标签，即默认保留 20%。不要误读参数名 `empty_think_ratio=0.2` 为“移除 20%”。
- `RLAIFDataset` 丢弃最后一个 assistant 占位，按 `thinking_ratio` 为 prompt 选择开关；PPO/GRPO 的 CLI 默认 0.9，覆盖 dataset 类本身 0.5 的默认值。
- Agent 每条完整轨迹抽一次开关，默认概率 0.1；这覆盖 `rollout_single` 函数签名中的 0.5 默认值。
- README 说明主线 SFT 已混入 Tool Call 和 reasoning 蒸馏数据；它同时承认 reasoning 与 Tool Call 联合样本不足，二者同时开启并不稳定。独立 `train_reason.py` 已移除。

来源：`model/tokenizer_config.json` 的 `chat_template`、`dataset/lm_dataset.py:29,195,208`、`train_agent.py:106,407`、[README thinking 说明](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md#L818)。

## 6. 后续启动前必须明确的兼容性和记录问题

1. **Windows Agent 工具实现不兼容。** `execute_tool` 使用 `signal.SIGALRM`/`signal.alarm`，且宽泛 except 会把失败变成 `None`。本机纯 CPU 检查确认这两个 API 都不存在。因此不作兼容修复时，模拟工具执行会全部走失败返回，不能称为正确复现 AgentRL。需要保持工具含义与超时约束的 Windows 实现，并用 CPU 测试证明执行和超时语义；本次审计未实施修复。来源：`train_agent.py:84`。
2. **Windows Agent DataLoader。** `collate_fn` 定义在脚本 `if __name__ == '__main__'` 块内，Windows spawn worker 无法依赖该主块重建同名函数；默认 workers8 存在 pickle/导入风险。workers0 是明确的本机兼容适配，或将 collate 提升为模块级函数再测试。来源：`train_agent.py:456,482`。
3. **不是换文件名前缀就能复制全部依赖。** DPO/PPO/GRPO/Agent 的 reference 必须保留原起点；PPO 恢复还必须包含 critic、critic optimizer 和两个 scheduler；白盒蒸馏必须另有匹配的 MoE SFT 教师。`init_model` 默认输入目录是 `../out`，多数训练入口不把 `args.save_dir` 传入它；PPO 的 critic 却直接从 `args.save_dir` 读同前缀文件，自定义 save_dir 时可能造成来源不一致。来源：`train_ppo.py:382,386`。
4. **尾部累积与恢复不能机械照搬。** DPO/LoRA/蒸馏/GRPO/Agent 在循环内保存，再在循环外处理不足一个累积周期的梯度；若为 8GB 适配增大 A，最终磁盘权重可能漏掉最后一次更新，周期保存也可能落在未完成累积的位置。GRPO 还在循环外直接引用 `step`，恢复到已完成 epoch 而 loader 为空时可能未定义。PPO 则在每个外层 rollout 后 flush 累积，但 `grad_accum_step` 不重置，改变 A 需独立检查计数语义。当前 pretrain/SFT 的本地修复不能视为这些其他脚本已修复。来源：`train_dpo.py:107,122`、`train_lora.py:60,71`、`train_distillation.py:119,137`、`train_grpo.py:176,198`、`train_agent.py:349,368`、`train_ppo.py:237,249`。
5. **现有 stdout parser 不覆盖所有后训练日志。** DPO 使用 `dpo_loss`、`learning_rate`；蒸馏使用 `ce/distill`；PPO 输出 Reward/KL/Critic Loss/Actor LR；GRPO 与 Agent 也有不同字段名。只匹配 pretrain/SFT 的 `loss/logits_loss/aux_loss/lr` 会让后训练曲线缺失。应按分支解析真实字段，区分 rollout step、optimizer step、reward 和训练 loss；没有验证集就不生成 val 曲线。
6. **8GB 可运行性尚未测量。** PPO 同时有 actor、critic、reference、1.8B RM；GRPO 默认一次 12 条最长约 1792-token 序列；Agent 训练打包默认 8 条、最多 2500 tokens；蒸馏同时驻留 Dense 学生和 FP32 MoE 教师。不能从文本 SFT 可运行推断这些默认配置都能装入 8GB，也不能悄悄删 RM、换教师、减小分组数后仍称“默认算法配方”。需要由后续实际显存探测决定并记录适配，禁止本报告虚构吞吐/耗时/质量。
7. **从 mini 基座继续后训练的结论边界。** README 的完整主线建议使用非 mini 数据。保留当前 mini pretrain，再复现同架构的所有后训练分支，可以准确称为“MiniMind-3 架构的官方 mini 基座与官方后训练分支复现”；它不等于已经复现公开完整训练权重或相同数据规模。
8. **Tokenizer 和 exam LoRA 不应混入必要串行链。** `train_tokenizer.py` 是词表训练示例，README 明确不建议重训当前 tokenizer；它不是 SFT 后的步骤。`minimind-3-exam` 是另一个 LoRA 格式对齐展示，README 说明其数据由 CEval/MMLU 的 test 子集抽样；若单独复现，必须核查评价集合重叠，不能把该对齐权重当作干净的通用能力提升证据。来源：`README.md:371,1614`。

## 7. 只读 inventory 与日志字段清单

每个分支至少独立登记这些公共配置：`script`、工作目录、`save_dir`、`save_weight`（LoRA 为 `lora_name`）、`from_weight`（蒸馏为两套前缀）、`data_path`、`epochs`、`batch_size`、`accumulation_steps`、`learning_rate`、`max_seq_len`、`dtype`、`num_workers`、模型尺寸/MoE 标志、`grad_clip`、`log_interval`、`save_interval`、`from_resume`、`use_compile`。输入权重、数据和 tokenizer 还应记录解析后的绝对路径与实际 hash，不能只记录短前缀。

| 分支 | 不能漏记的额外字段 / 默认值 |
|---|---|
| DPO | `beta=0.15`；policy/reference 各自输入路径和 hash（默认相同） |
| PPO | `critic_learning_rate=5e-7`、`clip_epsilon=0.2`、`vf_coef=0.5`、`kl_coef=0.02`、`gamma=1.0`、`lam=0.95`、`cliprange_value=0.2`、`ppo_update_iters=2`、`early_stop_kl=0.25`、`mini_batch_size=2`、`max_gen_len=1024`、`thinking_ratio=0.9`、`reward_model_path`、`rollout_engine=torch`；actor/reference/critic 起点、RM revision/hash |
| GRPO/CISPO | `loss_type=cispo`（GRPO 必须覆盖）、`num_generations=6`、`beta=0.1`、`epsilon=0.2`、`epsilon_high=5.0`、`max_gen_len=1024`、`thinking_ratio=0.9`、RM 与 rollout engine；policy/reference 起点 |
| AgentRL | 同上，但 `num_generations=4`、`max_gen_len=768`、`max_total_len=2500`、`thinking_ratio=0.1`；硬编码 `max_turns=3`、工具实现/超时适配版本、schema/GT、RM 默认仍必需 |
| LoRA | `lora_name=lora_medical`；实现常量 rank16、目标层筛选条件、仅 adapter 导出与完整 resume 文件的区别 |
| 蒸馏 | `student_hidden_size=768`、`student_num_layers=8`、`student_use_moe=0`、`from_student_weight=full_sft`；对应 teacher 值为 768/8/1/full_sft；`alpha=0.5`、`temperature=1.5`、两套权重 hash、共享 tokenizer hash |

三个 rollout 分支可选的 SGLang 字段是 `sglang_base_url=http://localhost:8998`、`sglang_model_path=../model`、`sglang_shared_path`（分别 `./sglang_ckpt_ppo`、`./sglang_ckpt_grpo`、`./sglang_ckpt_agent`）。默认使用 torch，不隐含启动或依赖一个现有 SGLang 服务。

所有训练日志的前缀都是 `Epoch:[epoch/epochs](step/iters)`。真实 stdout 字段（大小写/空格保留）如下：

| 分支 | stdout 字段 → 推荐归一化名 |
|---|---|
| DPO | `loss`、`dpo_loss`、`aux_loss`；`learning_rate` → `learning_rate`；`epoch_time` → `eta_minutes` |
| LoRA | `loss`、`logits_loss`、`aux_loss`；`lr` → `learning_rate`；`epoch_time` → `eta_minutes` |
| 蒸馏 | `loss`、`ce` → `ce_loss`、`aux_loss`、`distill` → `distill_loss`；`learning_rate`；`epoch_time` → `eta_minutes` |
| PPO | `Reward` → `reward`、`KL_ref` → `kl_ref`、`Approx KL` → `approx_kl`、`ClipFrac` → `clipfrac`、`Critic Loss` → `critic_loss`、`Avg Response Len` → `avg_response_len`、`Actor LR` → `actor_lr`、`Critic LR` → `critic_lr` |
| GRPO/CISPO | `Reward` → `reward`、`KL_ref` → `kl_ref`、`Adv Std` → `advantages_std`、`Adv Mean` → `advantages_mean`、`Actor Loss` → `policy_loss`、`Avg Response Len` → `avg_response_len`、`Learning Rate` → `learning_rate` |
| AgentRL | `Reward` → `reward`、`KL` → `kl_ref`、`GrpStd` → `group_reward_std`、`AdvStd` → `advantages_std`、`Loss` → `policy_loss`、`AvgLen` → `avg_response_len`、`AdvMean` → `advantages_mean`、`LR` → `learning_rate` |

PPO 的 stdout **没有 actor loss 字段**，不可从其他量补造。GRPO/Agent 日志的 `KL_ref`/`KL` 是有效 token 上 `ref_logp-policy_logp` 的带符号平均；它不等同于 loss 中使用的非负 KL 估计。`epoch_time` 在上述 CE/DPO/KD 日志中实际由剩余 step 估算而来，是 ETA，不能误标为已经消耗的 epoch wall time。DPO/LoRA/蒸馏横轴是数据 microstep；PPO/GRPO/Agent 是 rollout 数据批次 step，不能由日志本身捏造确切 optimizer-step 数。

本审计的可复现对象是固定提交中的脚本、配置、依赖和数据路径。README 的示例输出、曲线与 release 名称不构成每个发布 checkpoint 的完整训练履历。
