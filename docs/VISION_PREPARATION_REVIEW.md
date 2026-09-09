# 视觉数据准备只读审查

审查时间：2026-09-08 19:00（Asia/Shanghai）。范围仅为 `vision/scripts/prepare_evomind_v_data.py`、`vision/evomind_v/data.py` 的源数据提取，以及当前承接入口的就绪判断。未改动运行中的代码，未启动新的准备任务，未触及 GPU，未枚举图片目录内容或重算任何大文件哈希。本文不是重排/选样实现审查，也不授权视觉训练提前于文本后训练。

## 当前结论

**当前状态是 preparing，不是 ready。可以让现有生产进程继续运行，不能消费临时 manifest/parquet，不能启动第二份同目标提取。** 当前未发现 `evomind_split/summary.json` 或最终 `manifest.jsonl`。完成后可以复用本次产物，但必须经过下述就绪检查；不需要因为正在运行而重新提取。

2026-09-08 18:59:54 的只读文件快照：

| 对象 | 观察值 |
| --- | --- |
| 源 `sft_i2t.parquet` | 4,934,887,104 bytes；footer 为 2,904,511 行、581 row groups、Parquet 2.6 |
| 源字段 | `conversations`、`image_bytes` |
| `evomind_split/manifest.jsonl.tmp` | 740,140,839 bytes |
| `evomind_official_train.parquet.tmp` | 818,185,399 bytes |
| 最终 manifest / summary | 尚不存在 |
| E: 空闲空间（随后快照） | 705,487,261,696 bytes，约 657 GiB；不是本任务独占配额 |

随后进程快照显示：PID 23744 是 `.venv` launcher，PID 10232 是其实际 Python 子进程，两者不是两次独立提取。子进程工作集约 1.34 GiB；累计读/写计数约 7.77/2.24 GB。这是瞬时工作集和进程累计 I/O，不是峰值内存、已完成比例或 ETA。此前临时文件尺寸和写入计数均较小，现已增长；Windows 打开文件的 LastWriteTime/Length 可能滞后，不能据单次旧时间戳或零尺寸判定停滞。

实际命令为全源扫描：`--seed 42 --val-fraction 0.02 --test-fraction 0.02 --official-train-parquet dataset/evomind_official_train.parquet`，未设置 `--max-rows`。

已读取的小型已有 provenance：`artifacts/provenance/vision_assets.json` 记录源仓库 `jingyaogong/minimind-v_dataset`、revision `1e279a8b665cb10383451a6af6fd62b9f35bdd79`、上述文件大小和 SHA256 `712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643`。**这是既有验证记录的引用，本次未重新哈希源文件。**

## 完成提交与恢复边界

当前提交顺序（`data.py:234` 起）是：

1. 全部所选源行处理完，flush/fsync 临时 manifest，关闭 parquet writer。
2. `manifest.jsonl.tmp` 原子替换为 `manifest.jsonl`。
3. 官方 train-only parquet 的临时文件原子替换为最终文件。
4. 计算最终 manifest 和官方 parquet 的 SHA256，构造 `status: complete` summary。
5. summary 临时文件 flush/fsync 后，最后原子替换为 `summary.json`。

因此，**最终 summary 是完成标记；单独出现最终 manifest 或 parquet 不是完成标记。** 三个文件及图片目录不是一次原子事务。源文件哈希在数据扫描前计算；最终产物哈希在提交 summary 前计算，因而最后阶段即使没有新的图片写入也可能仍在工作。

当前没有恢复游标、已提交 batch 日志或 `--resume`：

- 临时 manifest 使用独占 `open("x")`；官方临时/最终 parquet 存在即拒绝；最终 manifest 或 summary 存在也拒绝重建。
- 中断可能遗留临时文件，也可能遗留“最终 manifest 已出现但 summary 尚未出现”的组合；再次原命令不会自动接着跑，而是 fail closed。
- 图片直接以独占 `xb` 写入最终内容寻址文件名，没有逐图临时文件替换或 fsync。异常退出可留下不完整图片；日后遇到同名既有图片会校验其完整 SHA256，并因不一致而停止，不会静默接受。
- 异常退出后的安全做法是先确认所有生产进程结束、保存日志和确切残留路径，再另行决定检查/修复或使用全新的输出目录。不要在活进程存在时删除、移动、改名任何临时产物；不要通过手动改名“补齐完成”。本文未执行任何恢复或清理。

这是一套防误覆盖、完成后可复用的流程，不是可从中断行续跑的流程。若未来实现恢复，应有源哈希/配置/代码版本绑定、batch 提交记录及对已写图片的处理策略，不能仅记录最后一行号。

## 防泄漏与数据语义

`stable_split`（`data.py:44`）用 `SHA256(seed + ':' + 原始图片字节SHA256)` 分桶。同一批输入中任何字节完全相同的图片，无论位于哪个文件或第几行，必定属于同一 split。当前 2% val + 2% test 意味着约 96% 图片哈希组进入 train，**不是保证恰好 96% 行数**。

官方 parquet 导出与规范化 manifest 在同一条已接受记录上判断 split：只导出接受的 train 记录，保留原 Arrow schema 和原始行内容，不受后续 20k pilot 限制。val/test 图片字节哈希组不会进入此官方训练导出。须明确：

- 全源扫描不等于所有源行均用于训练。多原图、不合规范对话、工具调用、非空 reasoning、无效图片等会拒绝并计数；真正训练集合是 accepted train rows。
- manifest 规范化角色并去除特定空 `<think>` 前缀；官方 parquet 保留原始对话。成员关系一致不代表序列化文本字节一致。
- 防泄漏只保证原始字节相同的图片分组。重新编码、缩放、裁剪的同图可能跨 split；未做感知去重审计，不能声称消除了所有视觉近重复。
- 对话/题目模板跨图片重复不被这套图片分组禁止。多次重复输入同一 source 也不会按对话去重，虽然图片仍不会跨 split。
- 源仅有上述两个字段，未提供 OCR/闭集真值类型，本次默认 `task_type=open`。末条回答可作为参考文本，但不能把普通描述 EM/CER 称为准确率。
- 提取阶段没有做 tokenizer/五视图长度筛选；`accepted` 不等于后续可训练或可生成的数量。共同五视图资格、pilot 选样及重排属于后续阶段，不在本文验证范围。

`iter_manifest`（`data.py:257`）会检查读到的记录版本、重复 sample_id、合法 split、同 hash 跨 split 冲突和对话格式，但不会主动要求 summary、验证 summary 哈希或按其 seed 重算 split。提前停止读取只验证已读前缀；它也没有限制解析后的图片路径必须留在 images 目录。生产器产生的路径是受控相对路径，但任意外来或被改过的 manifest 不能仅凭 iterator 通过就视为可信。

## 资源与性能边界

- Arrow 使用 32 行 batch；不把全部图片读入 Python 列表。但 `image_splits` 为所有唯一图片常驻字典，内存随 unique image 数增长；若提供 annotations，则整份 JSONL annotation 还会加载进内存。Arrow row-group 解压与 Python 转换也占内存，不能由 batch=32 推断恒定低内存。
- 源文件在流式扫描前额外完整读一遍计算 SHA256；重复图片每次遇到都会重新读取已落盘图片并核验哈希，可能增加大量重复 I/O。
- 同时保存内容寻址图片、文本 manifest、train-only parquet。parquet 中依然保留逐行原图字节，因此磁盘需求不能仅按原文件约 4.9 GB 或唯一图大小估算。代码当前没有磁盘余量门槛、总写入预算或定期进度日志。
- 每个有 train 记录的 32 行 batch 独立 `write_table`，可能形成约 9 万量级的小 row groups；具体数量必须完成后读取导出 footer，不能现在假定已发生的精确值。后续读取的 I/O/metadata 开销可能高于源文件的 581 组。
- 大量图片位于同一平面目录，会产生文件创建、目录元数据及安全软件扫描开销。当前不做全目录枚举，实际 unique image 数以最终 summary 为准。
- PIL `verify()` 是结构校验，不等于每幅图像全部像素解码成功。后续实际读取应保持哈希与 decode 失败显式报错，不能静默跳图而改变 A/B/C 样本集合。
- CPU 准备虽然不调用 CUDA，仍与文本训练共享磁盘、CPU、RAM。建议只运行这一份提取，观察既有训练步时与系统余量；不要因 GPU 空闲或文件暂时不增长启动额外全量检查。

## summary 就绪契约与必须追加的轻量检查

在最终 summary 出现且生产命令成功退出后，承接方应进行以下检查。若 summary 不存在，状态只能为 `preparing`（活进程）或 `incomplete`（已退出）；任何不一致均 `failed_validation`，不得进入训练。建议在独立校验入口落实，不修改本次仍在运行的生产代码。

1. **类型、配置、路径。** JSON 可完整解析；`format_version == 1`、`status == 'complete'`、`grouping == 'sha256_original_image_bytes'`；seed=42、val/test=.02/.02、`max_rows is null`。最终 manifest 和官方 parquet 为预期目录下的非空普通文件；`official_train_parquet` 必须解析到 `vision/dataset/evomind_official_train.parquet`，不能仅相信 summary 提供的任意路径。摘要字段是 64 位十六进制值，不能为空。
2. **源合同。** 本次必须恰好一个 source，路径与 pinned 资产一致，`size_bytes == 4934887104`、`total_rows == 2904511`，记录的 SHA256 与既有已验证 provenance 相同。轻量检查当前 stat 大小/mtime 与开工快照一致；这只是改变预警，不替代内容哈希。源文件需保持不可变，当前代码没有快照锁，也没有记录结束时 stat 来证明扫描期间未被替换。
3. **计数守恒。** 所有计数为非负整数，缺失的 Counter 字段按 0 处理；`scanned == accepted + sum(rejected_*)`；`accepted == train_records + val_records + test_records == sum(task_*)`；当前全源模式 `scanned == sum(sources.total_rows) == 2904511`；`unique_images == sum(unique_images_by_split)`，各 split unique 数不大于该 split records；`official_train_rows == train_records`，当前三 split 均应非空。正式生成 500 条 test 是否可行仍由后续共同资格检查决定。
4. **Parquet footer。** 仅读取最终官方导出的 footer，确认可打开、schema 与源 schema 一致、`num_rows == official_train_rows`，记录 row-group 数量和文件大小。不读取全部记录，不打开尚未完成的临时 parquet 来判断成功。
5. **有界 manifest/图片抽查。** 在生产结束后固定抽取少量首尾完整行（例如各 3 条，读取有上限，不扫描全文件），核验 JSON/schema/sample_id/source row 范围；按 summary seed/fractions 重算这些行的 split；路径解析必须落在 images 目录，文件存在。对这几个固定样本可核验内容哈希并完整 decode；这只能称抽查，不可称全量防泄漏证明或全图完整性审计。本次尚未执行抽查。
6. **已有内容校验仍需保留。** 当前 `scripts/evomind_pipeline_step.py:95` 的复用分支已经重算最终 manifest 和官方 train parquet SHA256。那是正式承接时一次性完整内容校验，不属于轻量检查，本次没有代跑；不要用上面几条抽查替代它，也不要在当前生产过程中再并行重复大文件哈希。若随后文件发生改变，要使 ready 失效。
7. **可追溯小文件。** 追加独立验证记录，保存验证时间、源 stat、summary 内容/摘要、footer 计数、抽查 sample_id、实际 argv、生产脚本和 `data.py` 的小文件 SHA256、进程退出状态。当前 summary 没有 producer 代码摘要、annotations 内容摘要或峰值/运行时间；这些不能在报告中假称已由 summary 证明。本次无 annotations；未来使用时必须绑定其哈希。

当前复用入口已有参数匹配和产物完整哈希校验，但尚未检查上述 `status/format/grouping`、源合同、计数守恒、预期官方路径和 parquet footer；直接调用 `iter_manifest` 更不会执行这些就绪检查。**应补的是承接验证，不是现在重写、重启或抢读生产任务。**

异常恢复、感知去重、重新组织 parquet row groups 及任何全量重提取均需另立操作与资源预算。当前安全动作是等待现有 CPU 准备完成，再运行有界验证；视觉训练仍遵守文本全后训练优先的总顺序。
