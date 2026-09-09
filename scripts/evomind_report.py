"""Build a truthful local report from saved results; absent results stay pending."""
from pathlib import Path
import json
import time
from evomind_posttrain import build_plan
from evomind_harness_eval import TASKS

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    branch_names = [b["name"] for b in build_plan()["branches"]]
    lines = ["# evomind 技术与实验报告", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "本报告由真实产物生成。待测不代表零分；训练中、smoke和最终结果严格区分。", "",
             "## 项目来源", "", "语言模型与视觉基线分别基于 MiniMind / MiniMind-V，保留原作者及许可证。",
             "当前优先交付验收合格的文本→视觉/视频主线；全图+2×2局部视图与冻结编码器特征缓存是之后的扩展，尚未完成效果验证。多尺度思想并非首创。", "",
             "## 数据与运行状态", ""]
    for mode in ("text", "vision"):
        data = read(ROOT / "artifacts" / "provenance" / f"{mode}_assets.json")
        lines.append(f"### {mode} 资源\n")
        if data:
            for item in data["assets"]:
                lines.append(f"- `{item['repo']}` / `{item.get('filename','model')}`，revision `{item['revision']}`，SHA256 `{item['sha256']}`；条数：{item.get('records','不适用/见数据划分报告')}。")
        else:
            lines.append("资源校验待完成。")
        lines.append("")
    state = read(ROOT / "artifacts/runs/text_official_mini_20260908/state.json")
    lines.extend(["### 文本训练\n", f"状态：{state.get('status') if state else '未启动'}；阶段：{state.get('current_stage') if state else '待定'}。", "",
                  "架构：768维、8层、6400词表，63,912,192参数。预训练和SFT各2epoch。", "",
                  "预训练B32×累积8；SFT B8×累积2（显存适配）。Windows workers=0。保留官方cosine和loss；修复保存时序，不声称逐位复现。", ""])
    for name in ("pretrain", "full_sft"):
        stage_state = (state or {}).get("stages", {}).get(name, {})
        attempt = stage_state.get("attempt")
        resume_record = stage_state.get("initial_resume_checkpoint")
        if resume_record:
            lines.append(f"- {name} 当前attempt={attempt}，从已核验检查点恢复，SHA256 `{resume_record['sha256']}`。历史与恢复后的重复microstep不累计；保留原始日志，不声称逐位一致续跑。")
        metric = ROOT / "artifacts/runs/text_official_mini_20260908/metrics" / f"{name}.jsonl"
        if metric.exists():
            rows = [json.loads(s) for s in metric.read_text(encoding="utf-8").splitlines() if s.strip()]
            if rows:
                last = rows[-1]
                lines.append(f"- {name} 最新观测：attempt {last.get('attempt', '未记录')}，epoch {last['epoch']}，microstep {last['microstep']}，当前microbatch loss={last['loss']}。不是验证集loss。")
                if attempt is not None and last.get("attempt") != attempt:
                    lines.append(f"- 当前attempt {attempt} 尚无新损失日志；上面的观测来自此前attempt，不能当作恢复后的进度。")
        curve = ROOT / "artifacts/runs/text_official_mini_20260908/curves" / f"{name}.png"
        if curve.exists():
            lines.extend(["", f"![{name} training]({curve.relative_to(ROOT).as_posix()})", ""])
    text_eval = read(ROOT / "artifacts/evaluation/text_official_mini/summary.json")
    lines.extend(["", "### 中文功能诊断\n"])
    if text_eval:
        lines.extend(["这是固定版本eval_llm.py的8条自动问答，保存原始回答；不是准确率测验。", "", "```json", json.dumps(text_eval["groups"],ensure_ascii=False,indent=2), "```"])
    else:
        lines.append("待文本SFT完成后运行官方8条自动问答。固定seed42、官方采样参数，思考开关单列；自编20题已取消。")
    scope = read(ROOT / "configs/text_alignment_scope.json")
    product = read(ROOT / "configs/product_pipeline.json")
    run_path = ROOT / product["run_dir"]
    eval_path = ROOT / product["eval_dir"]
    branch_names = [b["name"] for b in product["stages"]]
    posttrain = read(run_path / "state.json")
    post_eval = read(eval_path / "state.json")
    lines.extend(["", "## 已固定的产品主线（2026-09-09）", "",
        "文本预训练 → SFT → DPO → 验收选择 SFT/DPO → CISPO → 验收选择 → 图文 → 视频。",
        "这是 evomind 的交付路线，初始化关系不同于 MiniMind 默认的独立分支示例，不声称逐项复现官方产品配方。", "",
        "Agent-CISPO 保留为工具任务扩展，从验收选中的文本权重出发，不阻塞视觉；LoRA、独立 GRPO、蒸馏暂缓。PPO 和 Agent-GRPO 保持取消。",
        "保留全量 19,502 条 RLAIF、既定 epoch、奖励公式、全部日志/曲线/权重；不重启正在运行的 SFT。", "",
        f"产品队列状态：{posttrain.get('status') if posttrain else '等待 SFT 完成'}；选中权重：{posttrain.get('selected_name') if posttrain else '待验收'}。",
        "配置：configs/product_pipeline.json；执行：scripts/evomind_product.py。旧六分支配置仅保留为历史配方，不再自动调度。", "",
        "### 阶段验收（工程启发式，不是官方阈值）", "",
        "先完成官方 8 问（思考开/关）、8 个工具例子及全部 7 项中英文 harness 客观评测。",
        "SFT 非思考 8 问：EOS 至少 50%、平均 token repeat-4 不超过 50%；不合格时标记需修复，不强行靠 RL 掩盖。",
        "后续模型：每项客观准确率下降不超过 1 个百分点，EOS 不下降、复读不增加，且至少一项改善达到 0.1 个百分点，才晋级；否则保留父权重。",
        "这些筛查阈值不是统计显著性，也不能证明回答正确或视频理解可用；原始回答和全部指标保留。人工盲评按用户要求取消。", "",
        "| 阶段 | epoch | 初始化 | 状态 |", "|---|---:|---|---|"])
    for branch in product["stages"]:
        row = (posttrain or {}).get("branches", {}).get(branch["name"], {})
        initial = row.get("initial_name", "full_sft" if branch["name"] == "dpo" else "SFT/DPO 验收选择")
        lines.append(f"| {branch['name']} | {branch['options']['epochs']} | {initial} | {row.get('status', '待运行')} |")
    lines.extend(["", "### 视频与 Agent 的边界", "",
        "现有 MiniMind-V 单图/多视角流程保留，完成后只能标记图像阶段完成，不能标记整个视频项目完成。",
        "视频任务、来源/划分、帧采样、时间信息编码、视频 SFT 与验收仍待实现并固定数据合同。",
        "Agent 扩展需要明确工具与任务；通用数学工具训练不等于视频 Agent。", "",
        "### 客观评测协议", "",
        "ceval-valid、cmmlu、arc_easy、piqa、openbookqa、hellaswag、social_iqa；完整 split、任务默认 few-shot、固定 seed、原生模板与逐题输出。", ""])
    benchmark_preparation = read(ROOT / "artifacts/benchmarks/chinese_public_prepare/preparation.json")
    if benchmark_preparation:
        lines.append("已校验并准备完整公开题集：" + "；".join(
            f"{name} {row['rows']:,}题、{len(row['subjects'])}学科"
            for name, row in benchmark_preparation["datasets"].items()) + "。准备数据不等于跑出模型分数。")
    lines.extend(["", "### 文本分支统一结果\n",
        "单元格保留harness原始指标名称（acc/acc_norm等）；未运行记待测，不用旧自写协议填表。", "",
        "| 模型分支 | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)])
    for name in ("full_sft", *branch_names):
        row = (post_eval or {}).get("models", {}).get(name, {})
        cells = []
        for task in TASKS:
            key = f"harness_{task}"
            result_path = row.get(key, {}).get("result", {}).get("path")
            summary = read(Path(result_path)) if row.get(key, {}).get("status") == "completed" and result_path else None
            values = (summary or {}).get("scores", {}).get(task, {})
            scores = [f"{metric.split(',')[0]}={value*100:.2f}%" for metric, value in values.items()
                      if metric in ("acc,none", "acc_norm,none") and isinstance(value, (int, float))]
            cells.append(" / ".join(scores) if scores else "待测/见原始分学科结果" if summary else "待测")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "逐题预测、任务配置及原始回答保存在各分支目录。官方工具8例采用本地有界mock执行，不是广泛Agent能力准确率。", ""])
    rlaif_audit = read(ROOT / "artifacts/provenance/rlaif_official_template_audit.json")
    if rlaif_audit:
        lines.extend(["RLAIF全量19,502条、4种模板条件共78,008次渲染检查通过，未因此过滤数据。源第3987、5013、5534、12338行在移除尾部空占位后仍以assistant结束；官方RM wrapper会把该内容当末轮query。这是保留官方数据时的数据质量例外，不能因为程序能运行就忽略其语义风险。", ""])
    assets = read(ROOT / "artifacts/provenance/posttrain_assets.json")
    if assets:
        for item in assets["assets"]:
            if item.get("records"):
                lines.append(f"- 后训练 `{item['filename']}`：{item['records']:,}条，完整官方文件；SHA256 `{item['sha256']}`。")
        lines.extend(["", "已准备但暂缓的蒸馏配方采用单独下载的官方公开MoE教师 `full_sft_768_moe.pth`，不是本项目自训。文件及token映射已核验，严格加载和真实数值/显存仍需预检；来源与版本见 `artifacts/provenance/posttrain_assets.json`。", ""])
    if posttrain:
        for name, branch in posttrain.get("branches", {}).items():
            if branch.get("error"):
                lines.append(f"- `{name}`待处理失败：{branch['error']}")
            curve = Path(branch.get("full_directory", run_path / name / "full")) / "curves.png"
            if curve.exists():
                lines.extend(["", f"![{name} observed training]({curve.relative_to(ROOT).as_posix()})", ""])
    preparation = read(ROOT / "artifacts/runs/vision_cpu_prepare_20260908/state.json")
    lines.extend(["", "### 视觉CPU准备并行\n",
                  "用户已允许视觉代码与CPU数据准备和文本GPU训练并行；这不开放正式视觉训练。复用已有提取器，低优先级、有界内存审计/重排，不重复提取或减少数据校验。", "",
                  f"CPU衔接状态：{preparation.get('status') if preparation else '待启动'}。即使完成完整性审计、图片抽样hash检查与数据重排，也不代表模型质量或GPU缓存一致性通过。", ""])
    lines.extend(["", "## 视觉对照（正式训练待完整文本之后）\n", "A/B/C从同一个文本基座初始化，不使用已看过全图文数据的权重冒充无泄漏初始化。按原图分组划分train/val/test，B和C均为320视觉token。",
                  "", "| 版本 | 完成状态 | 质量指标 | 耗时/显存 |", "|---|---|---|---|"])
    for variant in ("A", "B", "C"):
        files = sorted((ROOT / "vision/artifacts").glob(f"**/{variant}_seed*/summary.json"))
        lines.append(f"| {variant} | {'已有产物，见下方明细' if files else '待测'} | 不预填收益 | 不预填收益 |")
    split = read(ROOT / "vision/dataset/evomind_split/summary.json")
    if split:
        lines.extend(["", "### 视觉数据实际划分\n", "```json", json.dumps({
            "counts": split["counts"], "unique_images_by_split": split["unique_images_by_split"],
            "manifest_sha256": split["manifest_sha256"], "grouping": split["grouping"]}, ensure_ascii=False, indent=2), "```",
            "", "官方参考使用全部接受的train图片组，2epoch；A/B/C在全量manifest固定seed哈希重排后筛选20,000条符合共同长度预算的训练对话、2epoch、B1×累积4、seq768。重排不改变图片分组归属。",
            "所有组均冻结视觉编码器；受控组仅训练projector和LLM首尾层。种子42/123/2026。共同五视图长度筛选，超长完整拒绝，不伪造EOS。", ""])
    else:
        lines.extend(["", "视觉图片分组数据正在准备，条数不预填。", ""])
    summaries = sorted((ROOT / "vision/artifacts/evaluation").glob("*/summary.json"))
    if summaries:
        for file in summaries:
            data = read(file)
            keys = ("samples", "split", "eos_rate", "mean_repeated_4gram_fraction", "closed_samples",
                    "closed_answer_exact_match", "ocr_samples", "ocr_character_error_rate", "mean_generation_seconds",
                    "peak_allocated_bytes", "inference_dtype", "training_seed", "human_evaluation_status")
            lines.extend(["", f"### {file.parent.name}\n", "```json",json.dumps({k:data.get(k) for k in keys},ensure_ascii=False,indent=2),"```"])
    aggregate = read(ROOT / "artifacts/evaluation/vision_aggregate.json")
    lines.extend(["", "### 三seed受控结果\n"])
    if aggregate:
        lines.append(f"对照完整性检查：{'通过' if aggregate['controlled_comparison_ready'] else '未完成或未通过，不能下收益结论'}。")
        lines.extend(["", "以下为独立训练seed的均值±样本标准差（ddof=1），不是置信区间。缺结果明确待测。", "",
                      "| 版本 | EOS结束率 | 四元组重复率 | 训练循环秒数 |", "|---|---|---|---|"])

        def measurement(value, percentage=False):
            if not value or value.get("mean") is None:
                return "待测"
            factor = 100 if percentage else 1
            unit = "%" if percentage else ""
            std = value.get("sample_std")
            spread = f" ± {factor * std:.2f}" if std is not None else ""
            pending = "，未齐" if value.get("status") != "complete" else ""
            return f"{factor * value['mean']:.2f}{spread}{unit} (n={value['n']}{pending})"

        for variant, row in aggregate["variant_statistics"].items():
            lines.append(f"| {variant} | {measurement(row['evaluation']['eos_rate'], True)} | {measurement(row['evaluation']['mean_repeated_4gram_fraction'], True)} | {measurement(row['train_runtime_seconds'])} |")
        for path in aggregate.get("plots", {}).get("files", []):
            path = Path(path)
            if path.is_file() and path.resolve().is_relative_to(ROOT):
                lines.extend(["", f"![{path.stem}]({path.resolve().relative_to(ROOT).as_posix()})", ""])
        lines.extend(["", "### 缓存的总成本\n", "```json", json.dumps(aggregate["cache_amortization"], ensure_ascii=False, indent=2), "```",
                      "", "训练循环计时包含验证/保存，模型与数据初始化不在该秒数内；不是端到端部署延迟。", "",
                      "### 官方示例原始回答\n", f"原始案例：`{aggregate['cases'].get('path') or '待生成'}`。不是挑选最高分的演示。",
                      f"盲评状态：`{aggregate['blind_evaluation']['status']}`。不要求人工评分。",
                      "只做官方六图描述示例，不将其当作500样本留出评测或质量提升证据。", ""])
    else:
        lines.append("等待A/B/C训练与相同测试集评测结果，不以尚未运行记零分。")
    parity = read(ROOT / "vision/artifacts/parity/real_cache.json")
    lines.extend(["", "### 真实GPU缓存数值检查\n",
                  "待执行，CPU回归通过不能替代本项。" if parity is None else
                  f"实际检查 {'通过' if parity.get('passed') else '失败'}；完整误差、阈值与环境见 `vision/artifacts/parity/real_cache.json`。", ""])
    lines.extend(["", "## 指标与边界\n",
                  "- EOS结束率：生成在预算内实际输出结束符的比例；不是回答正确率。",
                  "- distinct-2 / repeat-4：报告中注明字符或token口径、是否宏平均、短输出处理；不能据此断言语义质量。",
                  "- EM只用于明确有标准短答案的任务，CER只用于OCR；开放描述需要人工/独立评审，不能直接拿参考描述做精确匹配准确率。",
                  "- 缓存需验证输出、loss、投影层梯度及初始化一致，成本包含首次特征预计算与后续训练。",
                  "- 不同token数量、数据规模、训练预算的结果不得声称严格等算力。",
                  "- 人工盲评已取消；没有人工质量胜率，也不保证任何分支质量提高。", ""])
    # Include the versioned presentation contract on every regeneration.
    # This is documentation only: it must not schedule extra training/evaluation.
    alignment = ROOT / "docs/README_REPORT_ALIGNMENT.md"
    lines.extend(["", alignment.read_text(encoding="utf-8"), ""])
    lines.extend(["", "## 评测恢复与数据加载口径", "",
        "2026-09-09 的旧评测遇到 C-Eval 网络中断、CMMLU 自定义脚本信任要求，以及 torch.dtype 结果序列化错误。旧日志保留，不计为有效成绩。",
        "harness v2 直接读取已校验的官方中文原始 CSV 压缩包，保持原生任务模板、few-shot 默认值和评分逻辑；没有调用旧自写 MCQ 评测。数据版本和适配代码哈希进入 contract。",
        "真实原生任务数据预检覆盖 C-Eval 52 学科/1,346 题、CMMLU 67 学科/11,582 题；预检不是模型成绩。结果仍须通过完整推理与落盘校验。",
        "修复记录：`docs/EVALUATION_RECOVERY_20260909.md`；预检：`artifacts/benchmarks/harness_local_preflight.json`。", ""])
    target = ROOT / "EVOMIND_TECHNICAL_REPORT.md"
    target.write_text("\n".join(lines),encoding="utf-8")
    print(target)

if __name__ == "__main__":
    main()
