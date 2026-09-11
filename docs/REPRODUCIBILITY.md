# EvoMind 最小复现指南

本页提供当前文本代码的独立入口：CPU 小模型检查，以及 pretrain、SFT、DPO、GRPO、CISPO 的单阶段训练。仓库包含模型、分词器和训练代码；完整数据、奖励模型和训练权重需要另行准备。代码检查通过不代表完整训练完成，也不代表模型质量提升。

## 1. 环境与依赖

在独立 Python 3.10 环境中，先安装适合本机系统、驱动和设备的 PyTorch，再安装文本最小依赖：

```bash
python -m pip install -r requirements-minimal.txt
```

该文件固定 `transformers==4.57.6` 和 `datasets==3.6.0`，并列出直接使用的 NumPy、Requests。PyTorch 单独安装，避免绑定某台机器的 CUDA wheel。文件是依赖最小集合，不是完整环境锁文件；未验证所有版本组合，也未在全新环境中完成安装测试。

2026-09-11 的本地验证环境为 Windows、Python 3.10.20、PyTorch 2.13.0+cu130、Transformers 4.57.6、Datasets 3.6.0、NumPy 2.2.6、Requests 2.34.2。入口遵循本项目 Windows 环境的导入顺序：先 `datasets`，再 `torch`。

原始 `requirements.txt` 保留演示、评测、数据处理等更大范围的依赖。最小入口不要求 Web UI、OpenAI SDK、W&B、SwanLab、TRL、PEFT 或 SGLang。pretrain/SFT/DPO 的 `--use_wandb` 实际延迟导入 **SwanLab**，需另行安装；本文不启用它，RL 本地 runtime 则拒绝外部追踪。

GRPO/CISPO 额外安装本地奖励模型需要的依赖：

```bash
python -m pip install -r requirements-rl.txt
```

SentencePiece 0.2.1 来自已记录的远端奖励分词器兼容处理；本页未重新验证奖励模型加载或完整 RL 训练。

## 2. 无训练资产的检查

在仓库根目录运行：

```bash
python -B scripts/evomind_reproduce.py --help
python -B scripts/evomind_reproduce.py smoke
python -B scripts/evomind_reproduce.py smoke --use-moe
python -B -m unittest discover -s test -p "test_evomind_reproduce.py" -v
```

`smoke` 加载仓库内分词器，用随机 token 执行 1 层、hidden size 32、词表 64 的 CPU 前向、反向和一次内存中的 SGD 更新，检查 loss、梯度、参数有限性。它不读取训练数据，不下载模型，不写 checkpoint，也不运行正式 trainer。`--use-moe` 检查同一实现的 MoE 分支。

入口帮助、训练计划和 launcher 单元测试只需 Python 标准库。下面即使禁用 site-packages 也可输出计划：

```bash
python -B -S scripts/evomind_reproduce.py train grpo --dry-run
```

安装依赖后，可调用真实 trainer 的参数帮助：

```bash
python -B scripts/evomind_reproduce.py trainer-help pretrain
python -B scripts/evomind_reproduce.py trainer-help sft
python -B scripts/evomind_reproduce.py trainer-help dpo
python -B scripts/evomind_reproduce.py trainer-help grpo
```

## 3. 完整训练需要的资产

预训练/SFT 的固定版本数据可通过现有准备脚本显式下载并校验：

```bash
python -B scripts/evomind_prepare_assets.py text
```

该命令可能下载较大文件。检查生成的 `artifacts/provenance/text_assets.json`，保留来源、revision、行数和 SHA256。DPO/RLAIF 来自同一数据仓库与 revision，可单独下载：

```bash
python -c "from huggingface_hub import hf_hub_download; [hf_hub_download('jingyaogong/minimind_dataset', repo_type='dataset', revision='312afb4f76391145c6902f765bb51691c09a12f5', filename=name, local_dir='dataset') for name in ('dpo.jsonl', 'rlaif.jsonl')]"
```

以下为源代码固定的校验值。上面的下载 API 命令本身不会生成本项目来源收据，应在训练前校验并存档：

| 文件 | SHA256 |
| --- | --- |
| `pretrain_t2t_mini.jsonl` | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| `sft_t2t_mini.jsonl` | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |
| `dpo.jsonl` | `ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768` |
| `rlaif.jsonl` | `8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba` |

数据格式见 `dataset/lm_dataset.py`：预训练每行含 `text`；SFT/RLAIF 含 `conversations` 消息列表；DPO 含 `chosen`、`rejected` 两个消息列表。RLAIF 的 prompt 使用 `conversations[:-1]`，自备数据需保留末条 assistant 消息约定。

奖励模型为 `internlm/internlm2-1_8b-reward`，当前固定 revision 为 `25f3593492ab4625ce00fce8c5e67802d6e702ca`。需要时下载到仓库内：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('internlm/internlm2-1_8b-reward', revision='25f3593492ab4625ce00fce8c5e67802d6e702ca', local_dir='models/internlm2-1_8b-reward')"
```

runtime 使用 `local_files_only=True`，加载该目录及模型随附的 Python 实现（`trust_remote_code=True`）。记录模型 revision、文件 hashes、reward dtype/device。

## 4. 单阶段训练

`train` 默认仅打印 JSON 计划；`--dry-run` 显式表达同一行为。将它替换为 `--execute` 才会启动一个 trainer。`--` 后是 trainer 原有参数，其路径值相对**仓库根目录**解析；含空格的路径需加引号。入口使用当前 Python，固定 `trainer/` 工作目录，默认 `--device cpu --num_workers 0`，其余预算沿用 trainer 默认值。下面选择 `cuda:0`；按实际显存配置 batch 和梯度累积，并记录差异。

使用独立 `repro_` 前缀逐阶段复现 pretrain → SFT → DPO：

```bash
python -B scripts/evomind_reproduce.py train pretrain --dry-run -- --device cuda:0 --epochs 2 --batch_size 32 --accumulation_steps 8 --max_seq_len 340 --learning_rate 5e-4 --save_weight repro_pretrain
python -B scripts/evomind_reproduce.py train sft --dry-run -- --device cuda:0 --epochs 2 --batch_size 8 --accumulation_steps 2 --max_seq_len 768 --learning_rate 1e-5 --from_weight repro_pretrain --save_weight repro_sft
python -B scripts/evomind_reproduce.py train dpo --dry-run -- --device cuda:0 --from_weight repro_sft --save_weight repro_dpo --init_dir out
```

完成并评估父模型后，GRPO/CISPO 从同一个选定 DPO 权重分别初始化。下面对齐项目正式比较的输入名称 `out/dpo_768.pth`；若使用上面的独立复现权重，将两条命令的 `--from_weight dpo` 同时改成 `--from_weight repro_dpo`：

```bash
python -B scripts/evomind_reproduce.py train grpo --dry-run -- --device cuda:0 --from_weight dpo --init_dir out --save_weight repro_grpo --batch_size 1 --accumulation_steps 2 --reward_model_path models/internlm2-1_8b-reward --reward_device cuda:0 --reward_dtype float32
python -B scripts/evomind_reproduce.py train cispo --dry-run -- --device cuda:0 --from_weight dpo --init_dir out --save_weight repro_cispo --batch_size 1 --accumulation_steps 2 --reward_model_path models/internlm2-1_8b-reward --reward_device cuda:0 --reward_dtype float32
```

这两条命令保留同样的父权重、完整 RLAIF 数据、G=6、prompt 768、generation 1024、1 epoch、LR 3e-7 和 FP32 reward。`cuda:0` 是本机可见设备编号，不绑定历史服务器的物理 GPU。显存不足时需显式记录硬件适配；示例不构成新模型已通过质量选择的证据。

| 阶段 | 实际脚本与路径约定 |
| --- | --- |
| pretrain | `trainer/train_pretrain.py`；随机初始化；示例输出 `out/repro_pretrain_768.pth` |
| SFT | `trainer/train_full_sft.py`；读取 `out/repro_pretrain_768.pth`；输出 `out/repro_sft_768.pth` |
| DPO | `trainer/train_dpo.py`；读取 `--init_dir` 中的父权重；始终使用现有 `OfflineRuntime`，不需要也不支持 `--evomind_runtime` |
| GRPO/CISPO | `trainer/train_grpo.py`；入口加 `--evomind_runtime` 和明确的 `--loss_type grpo/cispo`；读取 `--init_dir` 中的指定父权重 |

文件名数字后缀取决于 `hidden_size`，MoE 另加 `_moe`。父子模型 hidden size、层数、MoE 和 tokenizer 必须兼容。预训练/SFT 的父权重读取固定在 `out/`，更改 `--save_dir` 不会更改父权重目录；这两个 trainer 不支持 `--init_dir`，恢复状态默认写入 `checkpoints/`。

入口执行前检查本地资产并拒绝已有输出名称；具体 dtype、shape、契约兼容性由 trainer 校验。预训练/SFT/DPO 使用 `--from_resume 1` 恢复，RL runtime 使用 `--resume` 指定 optimizer-boundary checkpoint。恢复前保留原配置与数据；RL checkpoint 还绑定源码哈希，代码变化后不会自动允许跨版本续训。DPO/RL 的有界集成实验可使用 `--max_steps` 和独立 `--save_dir`、`--resume_dir`；预训练/SFT 无此参数。

## 5. 日志、曲线与正式结果

直接 trainer 的控制台输出不会自动变成历史 supervisor 的曲线收据。执行时可以保存原始日志（以下是一次完整预训练调用，需先准备数据）：

```bash
python -u -B scripts/evomind_reproduce.py train pretrain --execute -- --device cuda:0 --epochs 2 --batch_size 32 --accumulation_steps 8 --max_seq_len 340 --learning_rate 5e-4 --save_weight repro_pretrain > repro_pretrain.log 2>&1
```

DPO 的 `OfflineRuntime` 以及 GRPO/CISPO runtime 另有结构化指标与恢复状态。保留它们和 checkpoint，不要仅保留截图或最终 loss。完整复现应归档 Git commit、命令、环境、数据/父权重 SHA256、原始指标、曲线、恢复收据、评测契约与逐条输出。

`configs/text_official_mini.json`、`configs/product_pipeline.json` 和远端脚本保存既有实验流程，涉及后续评测、父权重选择与图像/视频阶段，不是本页的最小入口。wrapper 不调用这些调度器，也不改变既有 config、loss、奖励函数或训练代码。本文验证没有启动完整训练。

`scripts/evomind_text_eval.py` 将输出绑定到权重、分词器、评测源码、解码参数、prompts 和 seed；不兼容的评测应使用新目录。自动分数、EOS、重复率仅说明各自测量的行为；没有相应协议与证据时，不据此声称事实可靠性、人类偏好、多模态能力或统计显著优势。

### 复现 README 的七项 harness 基准

这些基准由 `scripts/evomind_harness_eval.py` 执行。先安装可选评测依赖 `lm_eval==0.4.13` 的 HF extra；该 extra 还会安装 HF 后端所需的 Accelerate、PEFT 等包，因此评测依赖多于最小训练依赖：

```bash
python -m pip install -r requirements-minimal.txt "lm_eval[hf]==0.4.13"
python -B scripts/evomind_harness_eval.py --help
```

当前 CLI 只接受 `--checkpoint`、可选 `--lora`、单个 `--task` 和 `--output-dir`。它没有 `--limit` 或 `--device`：使用 CUDA 上的 Dense 768 维、8 层模型，FP16、batch=1、seed=42、chat template 开、thinking 关、完整 split；few-shot 使用任务默认值，在已发布结果中均为 0。这里的 `smoke` CPU 小模型无法作为该固定结构加载器的评测权重。

已有匹配结构的 checkpoint 后，可以先执行一个完整任务，例如 OpenBookQA 的 500 条 test 样本。这仍是正式评测调用，可能下载评测数据并使用 GPU：

```bash
python -B scripts/evomind_harness_eval.py --checkpoint out/repro_dpo_768.pth --task openbookqa --output-dir artifacts/evaluation/repro_dpo/harness_openbookqa
```

C-Eval 与 CMMLU 需要先准备固定版本的中文数据归档。下面命令只准备数据并记录校验信息；归档默认保存到后续 harness 所读取的 `artifacts/benchmarks/chinese_public/`：

```bash
python -B scripts/evomind_chinese_eval.py --prepare-only --output-dir artifacts/benchmarks/chinese_prepare_receipt
```

在仓库根目录执行以下 Python 代码，逐项评测一个 checkpoint 的全部七个任务。将 `checkpoint` 和输出目录中的 `repro_dpo` 一起改为其他待测模型，即可分别评测 SFT、CISPO 和 GRPO：

```python
import subprocess
import sys

checkpoint = "out/repro_dpo_768.pth"
tasks = ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa")
for task in tasks:
    subprocess.run([
        sys.executable, "-B", "scripts/evomind_harness_eval.py",
        "--checkpoint", checkpoint, "--task", task,
        "--output-dir", f"artifacts/evaluation/repro_dpo/harness_{task}",
    ], check=True)
```

每个任务目录保存 `contract.json`、含逐条输出的 `results.json` 和 `summary.json`。相同命令再次运行会验证并复用已完整保存的任务；未完整保存结果的任务会重新计算，不能承诺任务内部逐样本续跑。新权重或新合同使用新目录。

README 表格使用各任务（或任务组）原生 `acc,none × 100`，与 `acc_norm,none` 分开报告。复现历史数字还需要同一 checkpoint、分词器、任务数据及软件源码；只固定包版本不保证数值相同。已发布 SFT/DPO 与 CISPO/GRPO 的 harness 安装源码 hash 不同，完整合同和限制见[公开证据快照](results/EVIDENCE_20260911.md)及[基准 JSON](results/text_benchmarks_20260911.json)。本次仅校验这些命令与实际 argparse 接口，未运行评测或下载。

## 6. K=4 rollout 复用机制实验

`configs/ratio_reuse_ablation_k4_20260911.json` 复用同一 rollout 做四次 actor 更新，以观察 ratio 偏离 1 后 GRPO clipping 与 CISPO detached cap 的介入条件。它不能直接证明通用能力优势，也不能代替产品权重选择。相关检查命令为：

```bash
python -B -m unittest discover -s test -p "test_ratio_diagnostics.py" -v
```

如需重新运行该短对照，在完成第 4 节 DPO 后，先检查两份计划。两者使用同一个 `repro_dpo` 父权重、各 120 次优化器更新、K=4、LR=1e-6、batch=1、累积=1；添加 `--execute` 替换 `--dry-run` 才会训练。输出使用独立名称，不能覆盖正式分支：

```bash
python -B scripts/evomind_reproduce.py train grpo --dry-run -- --device cuda:0 --from_weight repro_dpo --init_dir out --save_weight repro_grpo_k4 --batch_size 1 --accumulation_steps 1 --learning_rate 1e-6 --updates_per_rollout 4 --max_steps 120 --ratio_diagnostics --reward_model_path models/internlm2-1_8b-reward --reward_device cuda:0 --reward_dtype float32
python -B scripts/evomind_reproduce.py train cispo --dry-run -- --device cuda:0 --from_weight repro_dpo --init_dir out --save_weight repro_cispo_k4 --batch_size 1 --accumulation_steps 1 --learning_rate 1e-6 --updates_per_rollout 4 --max_steps 120 --ratio_diagnostics --reward_model_path models/internlm2-1_8b-reward --reward_device cuda:0 --reward_dtype float32
```

复画公开逐步数据时，先按[证据快照中的拆分代码与完整命令](results/EVIDENCE_20260911.md#曲线与-k4-机制实验)生成 `artifacts/release_replot/cispo.jsonl` 和 `grpo.jsonl`，再执行：

```bash
python -B scripts/summarize_ratio_reuse.py --cispo artifacts/release_replot/cispo.jsonl --grpo artifacts/release_replot/grpo.jsonl --output artifacts/release_replot/summary.json
python -B scripts/plot_ratio_reuse.py --cispo artifacts/release_replot/cispo.jsonl --grpo artifacts/release_replot/grpo.jsonl --output artifacts/release_replot/curves.png
```

绘图另需 Matplotlib；它不属于最小训练依赖。
