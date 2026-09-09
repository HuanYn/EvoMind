# evomind-v

**evomind 的视觉理解分支：单图基线、多视角输入与冻结特征缓存。**

[文本主分支](https://github.com/HuanYn/evomind/tree/main) · [数据处理](evomind_v/data.py) · [模型](evomind_v/model.py) · [训练](scripts/train_evomind_v.py)

## 目标与状态

evomind-v 为文本基座接入视觉编码器与可训练投影层，建立图像理解基线，再比较全图 + 局部视图以及特征缓存的效果与成本。

**2026-09-09：代码与数据准备已完成相应 CPU 检查，正式视觉训练和真实 GPU 缓存收益尚待验证。** 本分支尚不是已完成的视频模型，也没有可下载的最终视觉权重。

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

下面的命令运行数据、mask、多视图和缓存回归测试；测试通过不等于真实 GPU 效果通过：

```bash
python -B -m unittest discover -s test -v
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
