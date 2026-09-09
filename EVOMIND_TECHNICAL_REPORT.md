# evomind 技术与实验报告

生成时间：2026-09-09 15:57:19

本报告由真实产物生成。待测不代表零分；训练中、smoke和最终结果严格区分。

## 项目来源

语言模型与视觉基线分别基于 MiniMind / MiniMind-V，保留原作者及许可证。
当前优先交付验收合格的文本→视觉/视频主线；全图+2×2局部视图与冻结编码器特征缓存是之后的扩展，尚未完成效果验证。多尺度思想并非首创。

## 数据与运行状态

### text 资源

- `jingyaogong/minimind_dataset` / `pretrain_t2t_mini.jsonl`，revision `312afb4f76391145c6902f765bb51691c09a12f5`，SHA256 `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`；条数：1270238。
- `jingyaogong/minimind_dataset` / `sft_t2t_mini.jsonl`，revision `312afb4f76391145c6902f765bb51691c09a12f5`，SHA256 `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`；条数：905718。

### vision 资源

- `jingyaogong/minimind-v_dataset` / `sft_i2t.parquet`，revision `1e279a8b665cb10383451a6af6fd62b9f35bdd79`，SHA256 `712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643`；条数：不适用/见数据划分报告。
- `jingyaogong/siglip2-base-p32-256-ve` / `model`，revision `9465d1dc89db6bc6227c5b6b0e0ca9b940325d62`，SHA256 `c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d`；条数：不适用/见数据划分报告。

### 文本训练

状态：running；阶段：vision_and_report。

架构：768维、8层、6400词表，63,912,192参数。预训练和SFT各2epoch。

预训练B32×累积8；SFT B8×累积2（显存适配）。Windows workers=0。保留官方cosine和loss；修复保存时序，不声称逐位复现。

- pretrain 当前attempt=2，从已核验检查点恢复，SHA256 `2121a8b9aab3b8f06313fd756af954ff8c89e158a5c735724d1887cce3a6f9c7`。历史与恢复后的重复microstep不累计；保留原始日志，不声称逐位一致续跑。
- pretrain 最新观测：attempt 2，epoch 2，microstep 39695，当前microbatch loss=1.6464。不是验证集loss。

![pretrain training](artifacts/runs/text_official_mini_20260908/curves/pretrain.png)

- full_sft 最新观测：attempt 1，epoch 2，microstep 113215，当前microbatch loss=1.7177。不是验证集loss。

![full_sft training](artifacts/runs/text_official_mini_20260908/curves/full_sft.png)


### 中文功能诊断

这是固定版本eval_llm.py的8条自动问答，保存原始回答；不是准确率测验。

```json
{
  "sampled": {
    "records": 8,
    "eos": 1.0,
    "eos_n": 8,
    "distinct_2": 0.6786696848368768,
    "distinct_2_n": 8,
    "repeat_4": 0.14724224896483798,
    "repeat_4_n": 8,
    "strict_exact_match": null,
    "strict_exact_match_n": 0,
    "elapsed_seconds": 5.6599096874997485,
    "elapsed_seconds_n": 8,
    "completion_tokens": 396.0,
    "completion_tokens_n": 8
  }
}
```

## 已固定的产品主线（2026-09-09）

文本预训练 → SFT → DPO → 验收选择 SFT/DPO → CISPO → 验收选择 → 图文 → 视频。
这是 evomind 的交付路线，初始化关系不同于 MiniMind 默认的独立分支示例，不声称逐项复现官方产品配方。

Agent-CISPO 保留为工具任务扩展，从验收选中的文本权重出发，不阻塞视觉；LoRA、独立 GRPO、蒸馏暂缓。PPO 和 Agent-GRPO 保持取消。
保留全量 19,502 条 RLAIF、既定 epoch、奖励公式、全部日志/曲线/权重；不重启正在运行的 SFT。

产品队列状态：incomplete；选中权重：full_sft。
配置：configs/product_pipeline.json；执行：scripts/evomind_product.py。旧六分支配置仅保留为历史配方，不再自动调度。

### 阶段验收（工程启发式，不是官方阈值）

先完成官方 8 问（思考开/关）、8 个工具例子及全部 7 项中英文 harness 客观评测。
SFT 非思考 8 问：EOS 至少 50%、平均 token repeat-4 不超过 50%；不合格时标记需修复，不强行靠 RL 掩盖。
后续模型：每项客观准确率下降不超过 1 个百分点，EOS 不下降、复读不增加，且至少一项改善达到 0.1 个百分点，才晋级；否则保留父权重。
这些筛查阈值不是统计显著性，也不能证明回答正确或视频理解可用；原始回答和全部指标保留。人工盲评按用户要求取消。

| 阶段 | epoch | 初始化 | 状态 |
|---|---:|---|---|
| dpo | 1 | full_sft | 待运行 |
| cispo | 1 | SFT/DPO 验收选择 | 待运行 |

### 视频与 Agent 的边界

现有 MiniMind-V 单图/多视角流程保留，完成后只能标记图像阶段完成，不能标记整个视频项目完成。
视频任务、来源/划分、帧采样、时间信息编码、视频 SFT 与验收仍待实现并固定数据合同。
Agent 扩展需要明确工具与任务；通用数学工具训练不等于视频 Agent。

### 客观评测协议

ceval-valid、cmmlu、arc_easy、piqa、openbookqa、hellaswag、social_iqa；完整 split、任务默认 few-shot、固定 seed、原生模板与逐题输出。

已校验并准备完整公开题集：ceval 1,346题、52学科；cmmlu 11,582题、67学科。准备数据不等于跑出模型分数。

### 文本分支统一结果

单元格保留harness原始指标名称（acc/acc_norm等）；未运行记待测，不用旧自写协议填表。

| 模型分支 | ceval-valid | cmmlu | arc_easy | piqa | openbookqa | hellaswag | social_iqa |
|---|---|---|---|---|---|---|---|
| full_sft | acc=22.81% / acc_norm=22.81% | acc=25.02% / acc_norm=25.02% | acc=29.92% / acc_norm=31.06% | acc=54.08% / acc_norm=52.45% | 待测 | acc=26.98% / acc_norm=27.58% | acc=34.65% |
| dpo | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| cispo | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

逐题预测、任务配置及原始回答保存在各分支目录。官方工具8例采用本地有界mock执行，不是广泛Agent能力准确率。

RLAIF全量19,502条、4种模板条件共78,008次渲染检查通过，未因此过滤数据。源第3987、5013、5534、12338行在移除尾部空占位后仍以assistant结束；官方RM wrapper会把该内容当末轮query。这是保留官方数据时的数据质量例外，不能因为程序能运行就忽略其语义风险。

- 后训练 `dpo.jsonl`：17,166条，完整官方文件；SHA256 `ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768`。
- 后训练 `rlaif.jsonl`：19,502条，完整官方文件；SHA256 `8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba`。
- 后训练 `agent_rl.jsonl`：39,988条，完整官方文件；SHA256 `cb96bcc8096aecc5eccaab858f75d5ace1dc22da2302c2230457e29744a761ab`。
- 后训练 `lora_medical.jsonl`：25,276条，完整官方文件；SHA256 `abf66d2bf14bf5704f6c9f1a166061f55d95e4dfd03c29a3c33d085ccaca593f`。

已准备但暂缓的蒸馏配方采用单独下载的官方公开MoE教师 `full_sft_768_moe.pth`，不是本项目自训。文件及token映射已核验，严格加载和真实数值/显存仍需预检；来源与版本见 `artifacts/provenance/posttrain_assets.json`。


### 视觉CPU准备并行

用户已允许视觉代码与CPU数据准备和文本GPU训练并行；这不开放正式视觉训练。复用已有提取器，低优先级、有界内存审计/重排，不重复提取或减少数据校验。

CPU衔接状态：complete。即使完成完整性审计、图片抽样hash检查与数据重排，也不代表模型质量或GPU缓存一致性通过。


## 视觉对照（正式训练待完整文本之后）

A/B/C从同一个文本基座初始化，不使用已看过全图文数据的权重冒充无泄漏初始化。按原图分组划分train/val/test，B和C均为320视觉token。

| 版本 | 完成状态 | 质量指标 | 耗时/显存 |
|---|---|---|---|
| A | 待测 | 不预填收益 | 不预填收益 |
| B | 待测 | 不预填收益 | 不预填收益 |
| C | 待测 | 不预填收益 | 不预填收益 |

### 视觉数据实际划分

```json
{
  "counts": {
    "scanned": 2904511,
    "accepted": 2654826,
    "train_records": 2544979,
    "task_open": 2654826,
    "rejected_image_outside_user_turn": 19422,
    "val_records": 54362,
    "test_records": 55485,
    "rejected_expected_one_original_image_marker": 228755,
    "rejected_unsupported_nonempty_reasoning": 2,
    "rejected_unsupported_tools_or_reasoning": 1503,
    "rejected_empty_answer": 3
  },
  "unique_images_by_split": {
    "train": 619092,
    "val": 13098,
    "test": 13036
  },
  "manifest_sha256": "91dbdf95a9f11a7c0c2d87f789d7e631504e8d0213470cb025866561e932e6d5",
  "grouping": "sha256_original_image_bytes"
}
```

官方参考使用全部接受的train图片组，2epoch；A/B/C在全量manifest固定seed哈希重排后筛选20,000条符合共同长度预算的训练对话、2epoch、B1×累积4、seq768。重排不改变图片分组归属。
所有组均冻结视觉编码器；受控组仅训练projector和LLM首尾层。种子42/123/2026。共同五视图长度筛选，超长完整拒绝，不伪造EOS。


### 三seed受控结果

对照完整性检查：未完成或未通过，不能下收益结论。

以下为独立训练seed的均值±样本标准差（ddof=1），不是置信区间。缺结果明确待测。

| 版本 | EOS结束率 | 四元组重复率 | 训练循环秒数 |
|---|---|---|---|
| A | 待测 | 待测 | 待测 |
| B | 待测 | 待测 | 待测 |
| C | 待测 | 待测 | 待测 |

### 缓存的总成本

```json
{
  "status": "pending",
  "preparation_records": [],
  "comparisons": [],
  "runtime_definition": "Recorded per-process training-loop wall time, including validation/checkpoint work; model/data initialization excluded. Not pure kernel throughput."
}
```

训练循环计时包含验证/保存，模型与数据初始化不在该秒数内；不是端到端部署延迟。

### 官方示例原始回答

原始案例：`待生成`。不是挑选最高分的演示。
盲评状态：`pending`。不要求人工评分。
只做官方六图描述示例，不将其当作500样本留出评测或质量提升证据。


### 真实GPU缓存数值检查

待执行，CPU回归通过不能替代本项。


## 指标与边界

- EOS结束率：生成在预算内实际输出结束符的比例；不是回答正确率。
- distinct-2 / repeat-4：报告中注明字符或token口径、是否宏平均、短输出处理；不能据此断言语义质量。
- EM只用于明确有标准短答案的任务，CER只用于OCR；开放描述需要人工/独立评审，不能直接拿参考描述做精确匹配准确率。
- 缓存需验证输出、loss、投影层梯度及初始化一致，成本包含首次特征预计算与后续训练。
- 不同token数量、数据规模、训练预算的结果不得声称严格等算力。
- 人工盲评已取消；没有人工质量胜率，也不保证任何分支质量提高。


## MiniMind-3 展示对齐清单

本节是 evomind 的报告展示合同，不是额外训练计划。参考固定版本 [MiniMind-3 README](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/README.md)。官方成绩、图片和经验耗时不作为 evomind 的实测结果。

### 图表与结果展示

| 官方展示 | evomind 对应展示 | 产物/边界 |
|---|---|---|
| 数据路线图与模型配置表 | 数据来源、revision、SHA256、条数、epoch、实际配置及权重继承关系 | 见本报告数据与产品主线；不能把官方独立分支画成本项目实际执行链 |
| 预训练曲线 | 语言预测 loss、学习率；MoE 才有辅助损失 | 已有实际曲线见文本训练；不得用零值代替未记录的字段 |
| SFT 曲线 | 总 loss、语言预测 loss、辅助损失的适用口径 | 训练中曲线保留；当前 batch loss 不标成 val loss |
| GRPO/Agent RL 曲线 | 当前主线 CISPO 的 reward、回答长度、学习率、优势统计、policy loss、参考差异等已记录字段 | 待 CISPO 训练；标明算法为 CISPO，不冒充官方 GRPO 实验 |
| DPO（官方无独立曲线图） | 保留本项目已记录的 DPO 指标和曲线 | 待训练；不因官方未画而删除本项目证据 |
| 七项客观成绩表与雷达图 | SFT、DPO、CISPO 的七项 harness 结果 | 表格待实测；雷达图待同口径数据齐全，缺值不填零、不插值 |
| 同题生成回答对比 | 官方固定问题、thinking 开/关、各权重完整回答 | 待评测；附采样配置、seed、结束原因，不挑最好回答替代原始记录 |
| Agent 逐题工具任务与成功率 | 官方脚本 8 例的题目、预期、实际输出、执行结果 | 待评测；与 README 的 20 题演示不是同一题集，不能直接比较 60%/85% |
| 长度—PPL / YaRN 对照 | 待评测（未加入当前自动训练队列） | 必须注明横轴字符/token、分词器和窗口口径；不代表视频能力 |
| 时间与成本表 | 实测设备、dtype、batch/累积、耗时、显存及计时范围 | 最终耗时待阶段完成；本机运行不套用官方 3090 租卡价格 |
| AI 裁判样例评分 | 当前未启用，明确不提供分数 | 人工盲评已取消；也不自动新增付费模型裁判 |
| PPO 曲线、Agent 扩展、其他算法 | PPO 不适用；Agent-CISPO 暂缓；LoRA/独立 GRPO/蒸馏暂缓 | 不为补图恢复已取消或暂缓的分支 |

### 指标展示规范

- 客观任务固定为 `ceval-valid`、`cmmlu`、`arc_easy`、`piqa`、`openbookqa`、`hellaswag`、`social_iqa`，同时保留中英文。显示原生指标键（如 `acc,none` / `acc_norm,none`）、样本量、split、few-shot、模板、版本与 checkpoint hash。
- 每张曲线注明横轴是 microstep 还是 optimizer step、所属 attempt、平滑方法（未平滑则明示），保留原始 JSONL；恢复训练的重叠区间不当作额外训练量。
- `logits_loss` 是语言预测损失，`loss` 是否含辅助项以本次代码为准；辅助损失标明原始值还是乘系数后的值。不能仅凭字段名称混用不同算法的统计量。
- reward 是既定评分函数的得分，不是正确率；回答长度是实际生成 token 数。组内 reward 标准差与标准化后的 advantage 标准差分别报告，不能互相替代。
- KL 相关字段注明估计公式和参考模型；带符号的 log-ratio 或有限样本估计不能不加说明地解释为精确 KL。
- 雷达图使用明确的统一刻度，不以面积宣称综合胜出。误差条只来自真实重复运行或适用的统计估计，不能把官方标准误复制到本项目。
- EOS、复读率保留为已有工程诊断，不能当作正确性或官方榜单分数；不因此新增评测任务。完整回答用于检查事实错误和内容退化。
- 缺数据写“待评测”，取消写“不适用”，暂缓写“暂缓”；“已生成代码”“已有文件”“训练完成”“评测通过”分开陈述。

后续结果沿现有报告生成入口更新。本清单不更改 epoch、全量 19,502 条 RLAIF、奖励函数、GPU 队列或 SFT → DPO → 选择 → CISPO → 选择 → 视觉/视频路线。



## 评测恢复与数据加载口径

2026-09-09 的旧评测遇到 C-Eval 网络中断、CMMLU 自定义脚本信任要求，以及 torch.dtype 结果序列化错误。旧日志保留，不计为有效成绩。
harness v2 直接读取已校验的官方中文原始 CSV 压缩包，保持原生任务模板、few-shot 默认值和评分逻辑；没有调用旧自写 MCQ 评测。数据版本和适配代码哈希进入 contract。
真实原生任务数据预检覆盖 C-Eval 52 学科/1,346 题、CMMLU 67 学科/11,582 题；预检不是模型成绩。结果仍须通过完整推理与落盘校验。
修复记录：`docs/EVALUATION_RECOVERY_20260909.md`；预检：`artifacts/benchmarks/harness_local_preflight.json`。
