# MiniMind 3 文本后训练：本机预检

## 2026-09-08 后续实施状态（优先于下面的历史初审）

下面的“本轮未下载/尚未支持”描述保留为实施前审计记录，不是当前最终状态。现已完成7个独立后训练分支的接续脚本、全量官方DPO/RLAIF/Agent/medical-LoRA文件的来源哈希校验、官方公开MoE教师与完整token映射校验，以及Windows有界工具执行、独立初始化/恢复目录、2次optimizer-update预检和正式epoch运行入口。

RL已显式支持CPU FP32 RM，Agent逐轮EOS和工具观察mask已有CPU回归；教师使用明确记录的BF16 autocast本地适配。默认G、长度、数据、epoch不暗改。公开C-Eval val 1,346题与CMMLU test 11,582题已准备到E盘；这不是GPU正式评分。RLAIF全量模板检查19,502条×4条件通过，4条末轮assistant源数据语义风险保留见 `artifacts/provenance/rlaif_official_template_audit.json`。

**仍待执行：**当前完整预训练/SFT结束、各分支真实GPU/RM及长序列显存预检、全量正式训练和模型评测。两次短预检成功也不证明所有长样本都能放入8GiB显存。最新状态以 `EVOMIND_STATUS.md`、运行state/receipt和原始日志为准；不把历史待办当成新执行结果。

## 历史初审（实现前）

本次范围纠正：先完成用户要求的 MiniMind 3 文本训练及后训练分支，再进入 MiniMind-V。本文是本轮源码、文件和依赖元数据审计，不是运行结果或训练完成证明。

审计对象为 `E:/project/Learning/evomind`，官方文本源 pinned commit 为 `6fc918beb68a0d8c40452338df6319fe168014ba`。本轮未安装依赖、未下载或复制数据/模型、未启动服务、未执行模型生成的工具表达式、未进行 GPU 训练。仅用标准库检查模块路径与 Windows signal；该检查进程隐藏了 CUDA，未导入 torch 或奖励模型。旧 `E:/project/Learning/HappyLLM` 仅做路径、大小和 SHA256 读取。

## 结论与阻塞

现有目录不能直接自动执行所有官方后训练入口。Agent 的 Windows 超时实现、工具评测的任意表达式执行、缺失的基础/教师权重与数据、以及尚未验证的显存峰值，必须分别处理。依赖缺失不等于算法无法运行；64M 主模型也不等于整个 RL 训练可以装进 8 GiB。

| 项目 | 已确认事实 | 影响与下一步 |
|---|---|---|
| Agent 工具执行 | `trainer/train_agent.py:88` 使用 `signal.SIGALRM`，随后 `signal.alarm(1)`；本机标准库检查两者均不存在 | `execute_tool` 的宽泛 except 会让所有已注册工具返回 None。必须先实现 Windows 可用的受限执行路径，再进行 Agent rollout |
| 工具评测安全 | `scripts/eval_toolcall.py:30` 对模型产生的 expression 直接调用 `eval`，未禁用 builtins；`:99` 执行工具时没有沙箱或超时 | 不应直接运行现有自动/交互式工具评测；应改为有界 AST 算术解释器，禁止任意 Python、属性遍历、文件/进程/网络访问 |
| 训练侧算术也不是沙箱 | `trainer/train_agent.py:58` 虽设置空 builtins，但仍调用 Python eval 并暴露 math 对象 | 空 builtins 不构成安全隔离；仍需受限语法、输入长度/深度/数值规模上限及可中止执行 |
| OpenAI SDK | 本环境 `find_spec('openai')` 为 None；`eval_toolcall.py:13` 无条件导入 OpenAI | 即便指定本地 backend，当前评测入口也会在导入时失败。后续可局部调整为 API 分支延迟导入，或只在项目环境安装明确需要的依赖 |
| Windows import 次序 | Agent/GRPO/PPO/DPO/distillation 已先 import datasets 再 import torch；`eval_toolcall.py` 当前没有此前置顺序 | 工具评测修订时一并保持已有 Windows DLL 兼容约定 |
| Windows DataLoader | Agent 的 collate_fn 定义在 `__main__` 分支中，默认 num_workers=8（`:456`, `:483`） | Windows spawn 子进程不能依赖此函数在导入时可用；先用 num_workers=0，若需多进程再将函数移到模块顶层并独立验证 |
| 基础权重 | 本轮读取时 `out/full_sft_768.pth` 尚不存在 | 它是正在排程的全文本 SFT 产物；后训练必须等待其真正完成并记录 hash，不能以 pretrain 或旧 19M 权重替代 |
| 蒸馏教师 | 默认 teacher_use_moe=1，期望 `out/full_sft_768_moe.pth`，本轮不存在 | 默认 MoE→dense 蒸馏尚缺教师分支。不能将 teacher_use_moe 改成 0 并使用同一个 dense SFT 权重后仍称默认教师蒸馏 |
| RM 默认相对路径 | 从 trainer 目录启动，`../../internlm2-1_8b-reward` 解析为 `E:/project/Learning/internlm2-1_8b-reward`，不存在 | 可在后续运行时显式传入已存在的 HappyLLM RM 绝对路径；本轮没有复制或加载它 |
| 新仓库后训练数据 | `dataset/dpo.jsonl`、`dataset/rlaif.jsonl`、`dataset/agent_rl.jsonl` 本轮均不存在 | 旧项目存在原始 DPO/RLAIF 文件；Agent 数据尚需单独解决来源、版本和完整性。不要把旧 5k/15k processed 子集称作全量 |
| 输出/初始权重目录 | `trainer_utils.init_model:119` 默认从 `../out` 取权重，部分调用未传 args.save_dir；PPO critic 又从 args.save_dir 取基础权重（`:385` 附近） | 自定义输出目录时可能出现 actor/ref 与 critic 搜索路径不一致。执行计划须明确 cwd、初始化目录和输出目录，必要时拆分配置 |

这些脚本没有现成的 `--max_steps` probe 开关。安全短试需要单独 harness 或明确记录的控制参数补丁；不能把启动后强杀当作成功训练或安全 checkpoint。

## 本机依赖证据

项目解释器：`E:/project/Learning/evomind/.venv/Scripts/python.exe`。其 `pyvenv.cfg` 指向 `C:/Users/25438/anaconda3/envs/MLLM`，`include-system-site-packages=true`。因此安装行为需要限制在项目环境，不能修改共享基础环境。

以下是已存在的 distribution 元数据，并非本轮 GPU/import 兼容性测试：

| 包 | 已见版本 |
|---|---|
| torch | 2.13.0+cu130 |
| transformers | 4.57.6 |
| datasets | 3.6.0 |
| pyarrow | 25.0.1 |
| numpy | 2.2.6 |
| tokenizers | 0.22.2 |
| requests | 2.34.2 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| einops | 0.8.2 |
| sentencepiece | 0.1.99 |

`openai`、`sglang`、`triton`、`flash_attn`、`trl`、`swanlab` 在本解释器模块发现检查中均为 None。不能据此要求安装整份 requirements：当前 GRPO/PPO 为手写训练循环，不导入 TRL；默认 torch rollout 不需要 SGLang；use_compile=0 不主动启用编译；use_wandb=false 不需要 swanlab。OpenAI SDK 是现有工具评测脚本的确定导入阻塞。奖励模型经 trust_remote_code 加载后的额外依赖和新版 transformers 兼容性尚未通过模型加载测试。

## 分支规模与 8 GiB 显存

全部可行性仍为 **待实测**。下表是源码参数和驻留模型构成，不是峰值显存测量。

| 分支 | 官方默认构成与规模 | 需要验证的重点 |
|---|---|---|
| DPO | policy + 冻结 reference；batch=4；chosen/rejected 合并，相当于 8 条序列；max_seq_len=1024；1 epoch | 策略反传、reference logits、偏好对长度及首次 Adam 状态初始化；可先 microbatch=1，记录累积调整 |
| GRPO/CISPO | policy + reference + FP16 RM；batch=2，G=6，prompt≤768，generation≤1024；1 epoch | 默认 12 条生成结果一起重算 policy/ref logits。先尝试 microbatch=1 而保留 G=6；不能只测生成就断言训练能跑 |
| PPO | actor + critic + reference + FP16 RM；batch=2，mini_batch=2，update_iters=2，prompt≤768，generation≤1024；1 epoch | 两个可训练网络和两个 optimizer；必须覆盖 critic、actor、第二轮 update、Adam 状态和保存。没有 RM 单独 CPU/offload 参数 |
| Agent RL | policy + reference + FP16 RM；batch=2，G=4，最多3轮，每轮 generation≤768，训练 max_total_len=2500；1 epoch | rollout 虽逐条顺序生成，但训练阶段会把 B×G=8 条结果 pad 后一起前向/反传；工具安全及 Windows 阻塞须先解除 |
| 默认蒸馏 | dense student + frozen MoE teacher；二者 hidden=768/layers=8；batch=32，seq=340；6 epochs | 缺 MoE SFT 教师；teacher 前向当前不在 student autocast 内；teacher logits/softmax 与 student 反传共存。需完成教师来源再 probe |
| MoE 分支 | `MiniMindConfig` 默认 4 experts、top-1；所有专家权重仍驻留 | 不能按 top-1 激活量估算权重/optimizer 显存；必须分别记录 MoE pretrain/SFT 的实际峰值和有效 batch |

本地 RM 两个 safetensors 文件合计 **3,399,182,888 bytes（约 3.17 GiB 文件体积）**。`LMForRewardModel` 在 `trainer_utils.py:180` 使用 FP16 AutoModel 并整体 `.to(device)`；GRPO/PPO/Agent 都直接把同一训练 device 传给它。这一文件体积不是 RM 运行峰值，还没有计入激活、KV、策略/参考/critic、梯度、optimizer、CUDA workspace 或桌面显存。

还存在一个实际影响显存的源码条件：`model/model_minimind.py:125` 仅在 mask 全 1 等条件下走 SDPA；有 padding 时进入显式 attention scores 和 softmax 路径。因此不同长度 prompt/多轮结果的 padding 会改变显存规模，短且等长的 smoke 不能代表真实长样本。

**不得将 G=1 当作等价省显存方案。** GRPO 和 Agent 均按组计算 `(reward - group_mean) / (group_std + eps)`，使用 unbiased=False；G=1 时 advantage 恒为0，只剩 KL 等项。若 microbatch=1 且保留官方 G 仍 OOM，可研究保持完整 reward group 的组内计算分块，或明确记录 G 改动为适配实验；均不可无记录地称为原默认复现。

建议分两级 probe：先短上下文覆盖 rollout→RM→reference→policy/critic backward→optimizer 的至少两个更新；通过后再覆盖官方 G、真实较长 prompt、生成上限和多轮长度。每阶段记录 allocated/reserved 峰值、实际 token 数、loss/reward/KL/梯度有限性、EOS/截断比例和首次 optimizer 分配。仅在现有 GPU 任务退出且有可用显存时执行；本轮未执行这些 probe。

## 工具行为与评测协议

官方 `eval_toolcall.py` 是交互演示，不能原样作为自动量化验收：只有 8 个固定示例，逐例重新随机 seed（`:231`），使用采样；`:179` 为没有总轮数上限的 while True，未输出固定测试集分数或完整审计 JSONL。API backend 默认指向 localhost，但可以由参数改为外部接口，本任务未授权外部费用。

工具结果也不是生产服务：训练中的时间固定为2025-03-07，天气/汇率/翻译来自小型查表；评测汇率固定7.15、翻译固定 hello world、单位换算一律乘0.621371。训练中的摄氏/华氏换算仅乘系数，缺少32度偏移。应将其标注为 mock 工具协议测试，不能把输出正确格式宣称为实时天气、汇率、任意翻译或通用换算能力。修订工具语义时还必须核查 Agent 数据 GT 是否沿用原 mock 语义，不能悄悄改变 reward 环境。

建议之后实施的协议：

1. 固定、本地、只读测试样本及 hash；每条保留实际对话上下文、工具 schema、允许的工具、参考结果和 max_turns/max_calls/max_new_tokens。对模型使用相同 seed/解码配置，单独报告 greedy 与采样。
2. 先独立测试解析器/执行器：非法 JSON、非对象 arguments、未知或本例未提供的工具、缺参/错类型、任意 Python/属性访问、超大表达式、死循环式调用都应返回明确错误；算术只解释白名单 AST，不运行 eval。未知单位/城市应明确报错或返回显式 mock 状态，不静默伪造有效结果。
3. 功能评测分别计数：JSON/schema 有效率、工具选择正确率、参数正确率、执行成功率、最终答案准确率（仅有可靠 GT 时）、调用次数、越权调用、超时、EOS/max-token/max-turn 终止和重复率。当前 `validate_gt_in_text` 是子串/数值命中启发式，不能替代完整答案正确性。
4. 对无 GT 的自由问答，报告停止/重复及原始样例；人工评分只导出空表。RM 分数是同一奖励模型的指标，不能单独证明通用质量提升。
5. Agent 当前从生成 token 中移除 EOS（`:123`），而后又对训练 mask 检测 EOS（`:286`）；多轮上下文还有生成侧未显式截断、训练侧左截断 max_total_len 的差异。需用带 EOS 和多轮 tool observation 的 tiny fixture 验证 token/logprob/mask 对齐，再讨论停止行为改善。

默认 torch rollout 可避免新增推理服务。若以后测试 SGLang：当前本机未安装；先核验 Windows/所选后端兼容性，不在本轮声称支持或不支持。`rollout_engine.py:125/185` 会发 `/generate` 与 `/update_weights_from_disk` 请求，更新路径来自共享目录；`:183` 使用 safe_serialization=False。示例以 `--host 0.0.0.0` 启动，但本任务只应考虑受控本地 loopback，不能照例对外暴露无鉴权权重更新接口。返回 token/logprob 数不匹配当前会补0/截断（`:144`附近），需把这种情况列为协议错误，不能静默用于RL比率。

## 可复用旧文件：已核验路径与 SHA256

以下仅证明本轮读取时文件存在及内容 hash；没有将其复制到 evomind、没有读取为训练 tensor、没有加载 RM，也没有用 processed 子集替代全量文件。

| 路径 | Bytes | SHA256 |
|---|---:|---|
| `E:/project/Learning/HappyLLM/data/raw/minimind_dpo/dpo.jsonl` | 53653322 | `ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768` |
| `E:/project/Learning/HappyLLM/data/raw/minimind_rlaif/rlaif.jsonl` | 23754740 | `8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/config.json` | 813 | `a50c39e11d22fb4250c2fa82724dbd59c8b2b1885fa85ca4565b4f710dc0c957` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/model-00001-of-00002.safetensors` | 1981392544 | `b0cbcdad5899d93dd653a4bf7fdf65e1d9668aaf61c0d5d41be1529e96b902ea` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/model-00002-of-00002.safetensors` | 1417790344 | `6421a9ec59b5e7b127c1b99fb80b88e5e013a9f2769a2f3000b8c99c99ad4630` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/model.safetensors.index.json` | 13682 | `d59589bef9aed26422d6475c4bba85aa8703197c50dbc0abf814a53434df3ffd` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/modeling_internlm2.py` | 91364 | `32114decf3b8e21af96ed5327f77d8cdefb2a96d26af4a1ed63af45f14572656` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/configuration_internlm2.py` | 9042 | `8b645f86e6a6bf9895eff775718fc85454da6c15c275bc28bf8fc54d1fc7f74c` |
| `E:/project/Learning/HappyLLM/models/internlm2-1_8b-reward/tokenization_internlm2.py` | 8806 | `444d4c2b0da158e61c34b3c727943f0ad454770c74b307f4d881f03603335eef` |

奖励模型加载使用 trust_remote_code=True；hash 并不代表这些 Python 文件已经完成安全审查或兼容性测试，后续加载前仍应审查其导入和执行内容。蒸馏只适用于已核对 token ID 对齐的教师/学生；代码裁切 teacher vocab 维度并不能使任意外部 tokenizer 自动对齐。

原始数据契约：DPO 使用 chosen/rejected 对话列表；RLAIF 使用 conversations，丢掉最后一个参考回答后生成；Agent 使用 conversations + gt，并从 system.tools 读取工具 schema。复用前需仅在 evomind 中记录原始来源/版本、schema 验收、数据划分与 hash；不要覆盖 HappyLLM 的既有结果。

## 后续门槛

范围与执行图首先应把文本分支全部列出，并区分共同 full_sft 起点的 DPO/PPO/GRPO/Agent 分叉、MoE 教师依赖，以及单独的 LoRA/蒸馏设置。本预检重点覆盖 Agent/GRPO/PPO/rollout/工具评测，附带 DPO、默认 MoE 教师的依赖风险；不是其他所有分支已经审计/跑通的声明。

每个分支的状态应依次记录为：依赖与数据待准备 → CPU 接口/安全测试通过 → GPU 短 probe 通过 → 完整训练进行中 → 完整训练与固定评测完成。代码修复、短 probe、后台启动均不等于全文本后训练完成。待上述范围与门槛落实后，再按最新用户要求安排 MiniMind-V。
