# README 架构与数据流图来源

这份说明记录 2026-09-12 README 三张文本 SVG 的事实依据。图由当前代码、已完成训练回执和数据来源清单整理；MiniMind 的 README 仅作为展示方式参考，其图、成绩、耗时不作为 EvoMind 的实测证据。

## Dense 与 MoE 架构

| 图 | 展示范围 | 可追溯实现与配置 |
|---|---|---|
| [Dense 架构](../figures/evomind_dense_architecture.svg) | 实际训练的 768 维、8 层 Dense 文本模型 | [MiniMindConfig、Attention、FeedForward、MiniMindBlock、MiniMindForCausalLM](../model/model_minimind.py)；[本轮预训练/SFT 配置](../configs/text_official_mini.json) |
| [MoE 架构](../figures/evomind_moe_architecture.svg) | 同骨干的可选 FFN 实现及公共权重准备 | 同一模型文件的 `MOEFeedForward`；[公开权重下载入口](../scripts/evomind_posttrain_assets.py) |

两者共用 hidden size 768、8 个 Decoder Block、6,400 词表、8 个 Q 头、4 个 KV 头、96 维 head、Pre-RMSNorm、Q/K Norm、RoPE 和共享的 Embedding/LM Head。Q/K Norm 与 RoPE 作用于 Q、K；每两个 Q 头共享一组 K/V，缓存保存未复制的四个 KV 头。默认 dropout 为 0。

Dense FFN 为 `down(SiLU(gate(x)) × up(x))`，gate/up 的 intermediate size 为 2,432。63,912,192 是共享权重去重后的可训练参数量，不把 Embedding 与 LM Head 重复计算；位置编码缓冲不计入参数。训练长度见下文阶段表，RoPE 配置支持的长度不等于已验证的长上下文能力。

MoE 每层有 4 个独立 SwiGLU 专家，每个专家的 intermediate size 同为 2,432。Router 对 token 做线性投影、softmax 和 Top-1 选择；`norm_topk_prob=true`，选中专家的归一化权重为 1。当前没有共享专家。负载均衡辅助项来自路由负载与概率，配置系数为 `5e-4`；辅助损失通过训练目标加入，不是另一条输出 token 主干。

当前匹配的公共 MoE 权重为 `models/teacher_official/full_sft_768_moe.pth`，来源 [jingyaogong/minimind-3-pytorch 固定版本](https://huggingface.co/jingyaogong/minimind-3-pytorch/tree/edba70ec15e06bc4280fbb96ac3383d73a7eab91)，SHA256 为 `a050020ea6d1b9e824693d0db525b1a0a8b40f36a934ea8d89da161368f20cc1`。它是上游公开 SFT 模型；本项目没有自行训练这条 MoE 预训练/SFT 轨迹，也没有用图示表示已经完成 Dense–MoE 质量对照。原始下载回执保存在本地 `artifacts/provenance/posttrain_assets.json`。

## 文本数据与训练关系

[文本数据流图](../figures/evomind_text_data_pipeline.svg) 的数据处理来自 [dataset/lm_dataset.py](../dataset/lm_dataset.py)，训练状态和参数来自 [text_training_20260911.json](results/text_training_20260911.json)，最终选择来自 [文本基座选择](TEXT_BASE_SELECTION.md)。

以下文件均来自 [jingyaogong/minimind_dataset 固定版本](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5)，revision 为 `312afb4f76391145c6902f765bb51691c09a12f5`。2026-09-12 对本地五个源文件重新统计行数并计算 SHA256，与准备阶段的 provenance 对照。这里的行数是 JSONL 源记录数，不是 token 数、去重问题数或通过长度过滤后的样本数。

| 文件 | 记录数 | SHA256 |
|---|---:|---|
| `pretrain_t2t_mini.jsonl` | 1,270,238 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| `sft_t2t_mini.jsonl` | 905,718 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |
| `dpo.jsonl` | 17,166 | `ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768` |
| `rlaif.jsonl` | 19,502 | `8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba` |
| `agent_rl.jsonl` | 39,988 | `cb96bcc8096aecc5eccaab858f75d5ace1dc22da2302c2230457e29744a761ab` |

准备入口为 [evomind_prepare_assets.py](../scripts/evomind_prepare_assets.py) 与 [evomind_posttrain_assets.py](../scripts/evomind_posttrain_assets.py)。原始回执为本地 `artifacts/provenance/text_assets.json`、`artifacts/provenance/posttrain_assets.json`；这两个运行产物不随普通 Git 提交公开，上表保留了复核源文件所需的版本和哈希。

| 阶段 | 输入构造与目标 | Epoch / 训练长度 | 实际初始化与状态 |
|---|---|---|---|
| Pretrain | `text` 经分词、BOS/EOS、截断、padding；非 padding 标签做 next-token CE | 2 / 340 | 从随机权重开始，已完成 |
| SFT | `conversations` 经聊天模板；assistant 回答及其终止标记参与 CE，其余标签忽略 | 2 / 768 | 从本轮 Pretrain 初始化，已完成 |
| DPO | chosen / rejected 分别模板化，在回答掩码上比较策略与冻结参考的 log-prob | 1 / 1,024 | 策略和参考都来自本轮 SFT，已完成 4,292 次更新 |
| CISPO | 对话前缀 → 在线生成 G=6 回答 → 奖励与组内优势 → policy loss 和 KL 项 | 1 / prompt 768、generation ≤1,024 | 从同一 DPO 初始化，已完成 9,751 次更新，选为 Dense 单图基座 |
| GRPO | 与 CISPO 匹配数据、生成预算、奖励与 KL 设置，policy loss 不同 | 1 / prompt 768、generation ≤1,024 | 从同一 DPO 初始化，已完成 9,751 次更新，保留为对照 |
| Agent-CISPO | 对话前缀、工具定义和 `gt` → 多轮调用 rollout | 参考配置 1 / prompt 1,024、generation ≤768、total ≤2,500 | 准备中，未训练；当前产品范围从选定文本 checkpoint 扩展 |

RLAIF 的 Dataset 用 `conversations[:-1]` 构造 prompt，并返回空 `answer`；源文件最后一条回答不作为此轮 RL 的监督答案。奖励模型为 [InternLM2-1.8B-Reward 固定版本](https://huggingface.co/internlm/internlm2-1_8b-reward/tree/25f3593492ab4625ce00fce8c5e67802d6e702ca)，实际两分支均为 BF16 policy/reference、同卡 FP32 reward。奖励由模型与规则项组成，训练 reward 不是回答正确率。

权重关系以完成回执为准：SFT 的 SHA256 是 `d99e5a37c6d61e2fbde64b595a2310751f35dfbb3fc66912de2f746e1c8eab8d`，DPO 为 `ccba444524c69d19a936646f296b89c13c3ee5bdc3f6cd30a037925a5b15ca2a`。CISPO 和 GRPO 的 `initial_checkpoint_sha256` 均等于这一 DPO 哈希；选中的 CISPO 最终权重为 `171d23ad98338540617532ca101c47045210203c798547e781141dcb77d0f1ed`。GRPO 不继承 CISPO，模型选择也不表示二者差异已达到统计显著。

[text_posttrain.json](../configs/text_posttrain.json) 保留早期独立 SFT 分支参考配方，其中 Agent-CISPO 的 1 epoch 等参数尚不是已执行记录。[product_pipeline.json](../configs/product_pipeline.json) 描述后来的分阶段产品选择，并将 Agent 定为从 `selected_text_checkpoint` 出发的延期扩展；其中早期 CPU reward 放置、GRPO 延期等字段不能覆盖已完成训练回执。README 采用实际运行关系，不把历史计划当成完成证据。

## 与视觉第一版的交接

文本图的视觉箭头表示已经验收的 CISPO 初始化进入 Dense 单图训练，不表示视觉训练或质量验收已经完成。第一版包含 Dense 单图，真正多图、视频和 MoE 视觉后续再推进。单图的五裁剪特征缓存依然来自同一张原图。

视觉模型、数据流与单图运行来源由 [evomind-v 技术报告](https://github.com/HuanYn/evomind/tree/evomind-v)维护。本机 `vision/` 是独立 checkout，不是 main 分支中可追踪的普通代码子目录。正文公开链接因此统一指向 `evomind-v` 分支。
