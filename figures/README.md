# EvoMind-V README 插图

视觉版沿用文本版的圆角浅色模块、柔和阴影和衬线斜体风格，图内描述当前 Dense 单图实现，不把后续多图、视频或 MoE 规划画成已完成能力。

| 图 | 矢量图 | 位图预览 | PDF | 可编辑源文件 |
|---|---|---|---|---|
| 单图 VLM 架构与冻结策略 | [SVG](evomind_v_architecture.svg) | [PNG](evomind_v_architecture.png) | [PDF](evomind_v_architecture.pdf) | [JSON](specs/evomind_v_architecture.json) |
| 视觉数据清洗、划分与训练 | [SVG](evomind_v_data_pipeline.svg) | [PNG](evomind_v_data_pipeline.png) | [PDF](evomind_v_data_pipeline.pdf) | [JSON](specs/evomind_v_data_pipeline.json) |

解释见 [README](../README.md)，版本、计数、处理逻辑和限制见 [数据来源说明](../docs/VISUAL_DATA_SOURCES.md)。图中训练长度为本轮配方，不是宣称模型永久上下文上限。中间语言层只冻结参数，反向梯度仍可穿过；最终六图是定性检查，不是全量视觉准确率。

五图共用文本分支的 [生成脚本](https://github.com/HuanYn/evomind/blob/main/scripts/render_readme_diagrams.py) 与 [重建说明](https://github.com/HuanYn/evomind/tree/main/figures)。本地 `vision/` 是独立 checkout；这两张图的源文件和导出文件由视觉分支单独管理。SVG 保留可编辑文字，PNG/PDF 用于固定预览。
