# evomind-v

**evomind 的视觉理解分支：单图基线、多视角输入与冻结特征缓存。**

[文本主分支](https://github.com/HuanYn/evomind/tree/main) · [数据处理](evomind_v/data.py) · [模型](evomind_v/model.py) · [训练](scripts/train_evomind_v.py)

## 目标与状态

evomind-v 为文本基座接入视觉编码器与可训练投影层，建立图像理解基线，再比较全图 + 局部视图以及特征缓存的效果与成本。

**2026-09-12：文本主分支已验收并选定 CISPO 基座，原生 Dense 单图训练已在 RTX 3090 上完成真实数据预检并开始正式更新。** 图像理解质量、多视图与缓存收益仍待评测；本分支尚不是已完成的视频模型，也没有可下载的最终视觉权重。

## 模型流程

```text
图像 → 单图 / 全图+2×2局部视图 → 冻结视觉编码器
                                      ↓
                                可训练 projector
                                      ↓
                          与文本 token 表示拼接 → LLM
```

缓存仅保存冻结视觉编码器输出，不缓存训练中的 projector 输出。单图与多视角使用不同 token 预算，不能宣称严格等算力比较。

## 实验设计

| 版本 | 输入 | 视觉编码器 | 状态 |
|---|---|---|---|
| A | 全图 | 在线、冻结 | 待正式训练 |
| B | 全图 + 4 个局部裁剪 | 在线、冻结 | 待正式训练 |
| C | 与 B 相同 | 预计算冻结特征 | CPU 一致性测试通过，GPU/效率待验证 |

A/B/C 从同一选定文本基座初始化。B/C 比较包括缓存准备成本，不仅比较后续训练耗时。尚不预填速度、显存或质量收益。

当前先运行下文的**原生单图基线**，不把它标成 A/B/C 已完成。其两步真实数据预检峰值 allocated 为 2,013.97 MiB，正式首步训练 CE 为 3.2035；这些数值证明训练入口可运行，不证明理解质量或缓存加速收益。

## 数据

图文资源来自 [minimind-v_dataset](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset) 的 `sft_i2t.parquet`。视觉编码器资源为 [siglip2-base-p32-256-ve](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve)。

本轮扫描 2,904,511 条对话，预处理接受 2,654,826 条。按原始图片字节哈希分组，避免同一图片跨训练、验证和测试划分。拒绝不支持的图片标记、空答案及部分工具/推理格式；不会在截断后补造 EOS。具体来源与配置由主分支 provenance 和视觉配置记录。

## 使用

在文本仓库根目录运行下面的命令，会把视觉分支放到流程预期的 `vision/` 目录；不会开始训练：

```bash
git clone --branch evomind-v --single-branch https://github.com/HuanYn/evomind.git vision
```

已有该目录时不要重复克隆。视觉训练由主分支的模型选择与配置控制，需先验收文本基座。

下面的命令在视觉目录查看训练参数，不会启动训练：

```bash
python scripts/train_evomind_v.py --help
```

### 原生单图基线

独立入口 [launch_dense_single.py](scripts/launch_dense_single.py) 从已验收文本权重运行原生 `train_sft_vlm.py`。先准备 `model/` 下的分词器、`model/siglip2-base-p32-256-ve/` 下的本地编码器，以及图像分组隔离后的 `dataset/evomind_official_train.parquet`。运行环境需要支持 BF16 的 CUDA PyTorch、Transformers 4.57.6、Datasets 3.6.0、PyArrow、Pillow 和 NumPy；启动器不会下载模型或数据。

在 `vision/` 目录执行，先将下列占位符替换为自己的验收文件、权重、数据哈希和已授权 GPU 的完整 UUID：

```bash
python scripts/launch_dense_single.py \
  --acceptance '<product_acceptance.json 路径>' \
  --init-weights '<验收选中的文本权重路径>' \
  --data dataset/evomind_official_train.parquet \
  --data-sha256 '<已核验的训练 Parquet SHA256>' \
  --run-dir ../artifacts/runs/dense_single_example \
  --gpu-uuid '<已授权且空闲的完整 GPU UUID>'
```

`--acceptance` 指定部署验收回执，状态必须为 `accepted_text_screening`；`--init-weights` 的 SHA256 必须与回执的 `selected.sha256` 一致。`--data` 与 `--data-sha256` 绑定完整训练文件。`--gpu-uuid` 只检查并绑定该卡，不因其他 GPU 上的任务而阻塞。`--run-dir` 必须是 evomind 项目内独立、初始为空的目录；外层 screen/nohup 日志应放在该目录之外。

固定配方为 768 维、8 层 Dense 文本模型，序列长度 768、BF16、学习率 `5e-6`、随机种子 42。`freeze_llm=1` 只训练 projector 与第 1、8 个解码器块，视觉编码器和其余语言模型参数冻结。训练集包含 2,544,979 条记录，完整遍历 2 epoch，共 5,089,958 次样本访问；有效 batch 为 4，共 1,272,490 次优化器更新（含每轮末尾的不足批次）。

启动器先以真实图文 batch 做两次预检更新，配置为 microbatch 4、累积 1；只有发生 CUDA OOM 才尝试 microbatch 1、累积 4。正式训练另起进程，从验收文本权重重新初始化，预检权重不进入正式训练。

运行目录保留 `contract.json`、验收回执副本、`state.json`、`probes/`、每次启动的 `logs/`、解析后的 `metrics.jsonl` 和 `curves.svg`，以及 `exports/`、原生恢复文件 `checkpoints/`、逐次保存的 `snapshots/` 与 `checkpoints.jsonl`。曲线中的 loss 是原生日志记录的微批次训练 loss。所有快照保留后会持续占用磁盘，需据实际保存频率检查剩余空间。`training_completed.json` 只标记两轮训练结束，六图描述等评测仍是独立步骤。

中断后在同一条命令末尾添加 `--resume`，输入、代码和配方必须与原运行合同一致。恢复读取最近保存的模型、优化器、scaler、epoch 和 microstep；原生检查点不保存数据增强随机数状态，因此不保证与不中断运行逐位一致，未保存的更新需要重算。正式训练已启动但尚无检查点时，启动器会停止并保留现场，不会自动从头重训。运行目录与 GPU 锁用于阻止重复启动。

下面的命令运行数据、mask、多视图和缓存回归测试；测试通过不等于真实 GPU 效果通过：

```bash
python -B -m unittest discover -s test -v
```

仅检查单图启动器与原生检查点衔接时，可运行以下 CPU 测试；集成测试使用三条合成样本并替换 GPU 调用，不需要真实模型、图文数据或 GPU：

```bash
python -B -m unittest discover -s test -p 'test_dense_single*.py' -v
```

## 目录

```text
evomind_v/    多视图、数据、模型与缓存
scripts/      数据准备、训练、评测、缓存校验
test/         数据完整性、恢复、缓存一致性测试
model/        基础模型组件
trainer/      基线训练入口
```

## 后续工作

- [ ] 正式图文基线训练与六图描述示例
- [ ] A/B/C 多随机种子训练
- [ ] GPU 数值一致性、显存与缓存端到端成本
- [ ] 视频数据合同、帧采样、时序建模和视频评测

多视角裁剪不等于视频；图文实验完成不能直接宣称视频项目完成。

## 许可

见 [LICENSE](LICENSE) 和 [NOTICE.md](NOTICE.md)。数据与外部权重遵守各自发布条件。
