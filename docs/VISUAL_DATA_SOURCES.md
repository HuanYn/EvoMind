# EvoMind-V 单图数据来源账本

本文对应 Dense 单图 V1 的固定数据快照。区分三类证据：发布者对上游构成的说明、本地文件与处理脚本可复验的事实，以及尚未核验的细节。所有计数均为数据准备结果，不是视觉理解评测成绩。

## 1. 直接输入与摘要

| 资源 | 固定版本/位置 | 本轮用途 |
|---|---|---|
| `jingyaogong/minimind-v_dataset` | revision `1e279a8b665cb10383451a6af6fd62b9f35bdd79` | 读取 `sft_i2t.parquet` |
| `jingyaogong/siglip2-base-p32-256-ve` | revision `9465d1dc89db6bc6227c5b6b0e0ca9b940325d62` | 冻结的图像编码器与预处理 |
| 已验收 CISPO Dense 文本权重 | 与文本验收回执 `selected.sha256` 绑定 | 语言模型初始化，不是视觉数据来源 |

```text
sft_i2t.parquet
  bytes  = 4,934,887,104
  rows   = 2,904,511
  SHA256 = 712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643

siglip2-base-p32-256-ve/model.safetensors
  SHA256 = c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d

siglip2-base-p32-256-ve/config.json
  SHA256 = 5ad8dda7d55541c7749f9b1cc43fe8eb8c70d8664588d89f710242ce06b3167e

siglip2-base-p32-256-ve/preprocessor_config.json
  SHA256 = d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff

本轮选定的文本初始化权重
  SHA256 = 171d23ad98338540617532ca101c47045210203c798547e781141dcb77d0f1ed
```

本地证据文件位于文本主仓的 `artifacts/provenance/vision_assets.json`，以及视觉仓的 `dataset/evomind_split/summary.json`。大文件、机器路径及运行产物不因本文存在而默认随 Git 仓库分发。

## 2. 上游数据是怎样来的

MiniMind-V 固定 commit `740d467ece78a0b7d2d976fcb424472095d4a688` 的 README 将该 SFT 包描述为下列混合。下表是**发布者的近似数量说明**，不是本项目按源类别重新清点的精确值：

| 上游构成说明 | 发布者说明的近似规模 |
|---|---:|
| ALLaVA-Instruct-LAION-4V，英文/中文 | 约 47 万 / 47 万 |
| ALLaVA-Instruct-VFLAN-4V，英文/中文 | 约 19 万 / 17 万 |
| LAION 指令增强：Gemini/Claude ensemble | 约 5 万 |
| LAION 指令增强：4o iterative | 约 5 万 |
| 纯文本对话，使用黑色占位图 | 约 23 万 |
| 混入的 caption 子集 | 约 127 万 |

Caption 部分同样来自 ALLaVA-4V 的 LAION/VFLAN 中英文集合。发布者描述的 LAION 图像偏自然场景，VFLAN 图像包含文档、图表和合成内容；图片在上游包装时被处理为约定的 256 分辨率 JPEG。本地模型仍按自己的 processor 做固定 256×256 输入预处理。

**没有另算一轮独立 `pretrain_i2t.parquet` 训练。** 当前直接读取 SFT 包，包中已经混入 caption 数据，不能再把 caption 的约 127 万条当作包外新增训练样本。

当前本地证据不能给出：每条数据完整的原始网页许可链、精确的上游子源比例、每条回答的教师模型/提示词/生成版本、逐条中英文标注与比例。子源名称含模型名并不等于本项目已经逐条核验其教师归属。可复验谱系从固定 Parquet 文件及其行号开始，向更上游追溯时明确依赖发布者说明。

## 3. 本地筛选与分组算法

实现：[prepare_evomind_v_data.py](../scripts/prepare_evomind_v_data.py)、[evomind_v/data.py](../evomind_v/data.py)。

1. 流式读取源 Arrow/Parquet 行，解析对话及其图像字节；记录来源文件索引和源行号。
2. 验证本期支持的对话角色、轮次顺序、非空 assistant 回答，以及恰好一个且位于 user 内容中的原始 `<image>` 标记。工具调用及不支持的推理格式不进入本期单图训练。
3. 对原始图片字节做图片文件结构校验，计算 SHA256，并按摘要存储图片。相同字节图片只存一份，对应的多条合规对话仍可保留。
4. 用图片摘要决定 train/val/test，使同图的所有对话落入同一分区。
5. manifest 保存规范化对话、sample ID、源行号、图片摘要和分区。原生训练 Parquet 另行导出接受的 train 组，**保留原始 Arrow 行内容**，不是把 manifest 文本重新合成训练样本。

分组计算可表示为：

```text
image_hash = SHA256(original_image_bytes)
u = int(SHA256("42:" + image_hash), 16) / 2^256

0.00 <= u < 0.02  → validation
0.02 <= u < 0.04  → test
0.04 <= u < 1.00  → train
```

sample ID 编码来源文件索引、原始行号和图片摘要前缀。这个身份标识用于追溯，不是额外的人工标签。

## 4. 精确计数

| 项目 | 行数 |
|---|---:|
| 扫描源行数 | 2,904,511 |
| 接受 | 2,654,826 |
| 拒绝 | 249,685 |
| Train | 2,544,979 |
| Validation | 54,362 |
| Test | 55,485 |

拒绝计数：图片标记数量不符 228,755；图片标记出现在非 user 轮次 19,422；不支持的 tools/reasoning 字段 1,503；非空 reasoning 内容 2；空答案 3。它们合计 249,685，不支持的格式不能全部解释成语义质量差。

接受集合共有 645,226 个原始图片字节哈希组：train 619,092、validation 13,098、test 13,036。所有接受记录的任务类型为 `open`；数据准备阶段没有为描述任务制造一个新的“客观准确率”标签。

```text
evomind_split/manifest.jsonl
  SHA256 = 91dbdf95a9f11a7c0c2d87f789d7e631504e8d0213470cb025866561e932e6d5

evomind_official_train.parquet
  rows   = 2,544,979
  SHA256 = 701a0da67ea63b7603092c78f912e89ed5ca4549659b4ffa2587a9d8762cab58
```

两轮训练访问 5,089,958 次样本。在每轮末尾保留不足批次、B4/A1 的配置下，优化器更新预算是 `2 × ceil(2,544,979 / 4) = 1,272,490`。

## 5. “已核验”与“尚未核验”

| 事项 | 证据范围 |
|---|---|
| 源文件版本、字节数、SHA256 | 已记录在资产与准备报告 |
| 源行数、拒绝理由、split 行数、图片组数 | 准备脚本全量计数 |
| 图片结构与哈希 | 准备阶段对图片字节处理；内容寻址存储 |
| 后续 manifest 完整性审计 | 全 manifest 结构/谱系检查；没有再次逐行回读源 Parquet 内容 |
| 后续图片摘要复查 | 确定性选取前 32 个不同图片，非随机抽样，也非重新穷尽核验全部图片 |
| 语义正确性、描述是否准确 | 没有逐条人工审核 |
| 近重复与污染 | 没有感知近重复、语义近重复或外部预训练污染的穷尽检查 |

后续审计依据是文本主仓的 `artifacts/provenance/vision_manifest_audit.json`；其中 `integrity_only=true`，`image_bytes_exhaustively_verified=false`。不能把该报告的 `pass` 扩大解释为所有图文内容的质量通过。

图片字节 SHA 相同的跨分区隔离是明确保证；视觉相同但重新编码、裁剪或缩放后的图片可能产生不同 SHA，这类近重复不由该机制解决。

## 6. 与控制实验及最终评测的关系

历史 `evomind_controlled/manifest.jsonl` 从相同接受集合出发，以 seed/sample ID 做确定性重排并调整相对图片路径。它服务旧 A/B/C 20k 控制实验，不能将其挑选规则、五视图 token 预算或权重结构替换到当前全量原生训练中。单图五裁剪也不是五张独立图片。

最终仍采用原定 `dataset/eval_images/` 六图的开放描述协议。保留的 validation/test 图像组不因此被宣称已经全量评测。六图输出保留原始文本和 token；EOS、重复率、耗时和显存分别描述生成行为及成本，不是视觉理解准确率。

## 7. 来源与许可

- 上游处理与数据组成说明：[MiniMind-V 固定 README](https://github.com/jingyaogong/minimind-v/blob/740d467ece78a0b7d2d976fcb424472095d4a688/README.md)。
- 本轮直接数据：[minimind-v_dataset 固定 revision](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/1e279a8b665cb10383451a6af6fd62b9f35bdd79)。
- 主要上游系列：[ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)。
- 直接视觉塔：[siglip2-base-p32-256-ve 固定 revision](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve/tree/9465d1dc89db6bc6227c5b6b0e0ca9b940325d62)，发布说明的源模型为 [Google SigLIP2](https://huggingface.co/google/siglip2-base-patch32-256)。

数据及外部权重适用各自的发布条件；代码许可见 [LICENSE](../LICENSE)，复用说明见 [NOTICE.md](../NOTICE.md)。本项目未将二次分发版本的许可证自动延伸为所有上游图像的许可证明。
