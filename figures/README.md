# EvoMind README 插图

五张插图统一采用浅色分区、圆角模块、轻阴影、衬线斜体标题和深色箭头，展示方式参考 MiniMind README；结构、数据量和完成状态依据本项目代码与运行记录重新绘制，不沿用上游实验成绩。

| 文本图 | 矢量图 | 位图预览 | PDF | 可编辑源文件 |
|---|---|---|---|---|
| Dense 主干、GQA、SwiGLU | [SVG](evomind_dense_architecture.svg) | [PNG](evomind_dense_architecture.png) | [PDF](evomind_dense_architecture.pdf) | [JSON](specs/evomind_dense_architecture.json) |
| MoE 路由与专家 | [SVG](evomind_moe_architecture.svg) | [PNG](evomind_moe_architecture.png) | [PDF](evomind_moe_architecture.pdf) | [JSON](specs/evomind_moe_architecture.json) |
| 文本数据与实际权重关系 | [SVG](evomind_text_data_pipeline.svg) | [PNG](evomind_text_data_pipeline.png) | [PDF](evomind_text_data_pipeline.pdf) | [JSON](specs/evomind_text_data_pipeline.json) |

另两张单图 VLM 架构、视觉数据流图位于 [evomind-v 分支](https://github.com/HuanYn/evomind/tree/evomind-v/figures)。逐图解释见 [主 README](../README.md)，事实依据见 [来源账本](../docs/README_DIAGRAM_SOURCES.md)。MoE 图展示支持的架构，不代表该分支已由本项目训练完成。

## 重新生成

[生成脚本](../scripts/render_readme_diagrams.py) 使用 FigureSpec JSON 生成 SVG；PNG/PDF 使用独立无界面 Edge 导出，不操作个人浏览器，不加载模型，也不占用训练 GPU。SVG 保留可编辑文字；不同系统的字体回退可能略有差异，固定预览可使用 PNG/PDF。

需要 Python、ARIS `figure-spec/scripts/figure_renderer.py`。后者是绘图工具依赖，不是训练依赖；浏览现有插图不需要安装它。若要同时重建五图，本地主仓库下需另有 `vision/` checkout。PowerShell 中设置渲染器实际路径，然后执行：

```powershell
# 目的：从已保存的 JSON 重建五张 SVG，不改训练文件。
$rendererPath = '填写实际路径/figure-spec/scripts/figure_renderer.py'
python scripts/render_readme_diagrams.py --renderer $rendererPath

# 目的：同时生成 PNG/PDF，并检查真实字体下的文本重叠、出界和框内溢出。
python scripts/render_readme_diagrams.py --renderer $rendererPath --edge 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
```

调整布局可直接编辑 `figures/specs/*.json`。若修改的是 Python 中的布局函数，添加 `--build-specs --vision-val 54362 --vision-test 55485`，会用代码布局重新生成五份 JSON。该选项会覆盖 JSON 的手动布局修改，应先保存自己的修改。

项目在标准 FigureSpec 之上扩展 `edges.via`（折线路径）、`edges.arrow`（是否显示箭头）、`panels`（浅色背景分区）。节点/连线和样式均在构建时确定性生成，不手改输出 SVG。验证结果保存在本机 `artifacts/readme_diagram_preview/*.layout.json`，浏览器临时目录也仅位于该产物目录。

## 检查

```powershell
# 目的：CPU 检查计数、训练谱系、图源一致性与公开资源齐备，不启动训练。
python -B -m unittest discover -s test -p test_readme_diagrams.py -v
```

除自动框检查外，交付前还逐图检查了箭头与残差含义。Dense 的第二条残差是第一次相加后的 `h`，不是再次加入原始 `x`；MoE 中路由选择与 token 特征输入分开画。GRPO/CISPO 的 G=6 不应用于图中尚未训练的 Agent 分支。
