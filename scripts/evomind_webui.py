"""EvoMind Studio: local Streamlit chat for strict native text checkpoints.

Run from the repository root: python -m streamlit run scripts/evomind_webui.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread

import datasets  # noqa: F401 -- Windows DLL import order
import streamlit as st
import torch

try:
    from transformers import TextIteratorStreamer
except ImportError:  # pragma: no cover - dependency-version dependent
    from transformers.generation.streamers import TextIteratorStreamer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evomind_load_text_model import load_model  # noqa: E402

SYSTEM_PROMPT = (
    "你是 EvoMind，由 YH 开发和维护的中文智能助手。"
    "当用户询问你是谁、由谁开发时，请回答：我是 EvoMind，由 YH 开发和维护的智能助手。"
    "请用自然、准确、简洁的中文提供帮助；不知道时如实说明。"
)

st.set_page_config(page_title="EvoMind Studio", page_icon="🧠", layout="wide")


def current_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def discovered_checkpoints() -> list[str]:
    paths: list[Path] = []
    for directory in (ROOT / "checkpoints", ROOT / "out"):
        if directory.is_dir():
            paths.extend(directory.glob("*.pth"))
    return [str(p) for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)]


@st.cache_resource(show_spinner="正在加载 EvoMind 权重……")
def load_native_model(checkpoint: str, use_moe: bool, device: str):
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"找不到有效 checkpoint：{path}")
    model, tokenizer = load_model(str(path), device=device, use_moe=use_moe)
    if device == "cuda":
        model = model.to(dtype=torch.float16)
    return model.eval().requires_grad_(False), tokenizer, str(path)


def make_prompt(tokenizer, messages: list[dict[str, str]], thinking: bool) -> str:
    dialogue = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    return tokenizer.apply_chat_template(
        dialogue, tokenize=False, add_generation_prompt=True, open_thinking=thinking,
    )


def display_text(raw_answer: str, thinking: bool) -> str:
    if thinking:
        return raw_answer
    if "</think>" in raw_answer:
        return raw_answer.rsplit("</think>", 1)[-1].strip()
    return raw_answer.replace("<think>", "")


def generate_stream(model, tokenizer, messages, *, thinking, temperature, top_p, max_new_tokens, max_context):
    prompt = make_prompt(tokenizer, messages, thinking)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context).to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0)
    options = {
        "input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5), "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True, "streamer": streamer,
    }

    def run_generation() -> None:
        with torch.inference_mode():
            model.generate(**options)

    worker = Thread(target=run_generation, daemon=True)
    worker.start()
    for fragment in streamer:
        yield fragment
    worker.join(timeout=1.0)


def sidebar() -> tuple[str, bool, str, float, float, int, int, bool]:
    with st.sidebar:
        st.header("运行设置")
        st.caption("加载当前 768×8 文本模型。")
        suggestions = discovered_checkpoints()
        checkpoint = st.text_input(
            "Checkpoint 路径", value=suggestions[0] if suggestions else "",
            placeholder="checkpoints/full_sft_768.pth",
        )
        use_moe = st.toggle("MoE 架构", value=False, help="需要结构匹配的 MoE 权重。")
        device = current_device()
        st.caption(f"运行设备：{device}")
        st.divider()
        thinking = st.toggle("显示思考模式", value=False)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        max_new_tokens = st.slider("最大生成 token", 32, 1024, 512, 32)
        max_context = st.select_slider(
            "Prompt 上下文窗口", options=[768, 1024, 1536, 2048], value=768,
            help="768 是当前主线 SFT 长度；更长选项用于推理外推测试。",
        )
        if max_context > 768:
            st.warning("实验模式：更长输入尚未经过长上下文专项训练与评测。")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pending_response", None)
            st.rerun()
    return checkpoint, use_moe, device, temperature, top_p, max_new_tokens, max_context, thinking


def render_pending_response(checkpoint, use_moe, device, temperature, top_p, max_new_tokens, max_context, thinking) -> None:
    if not st.session_state.pop("pending_response", False):
        return
    try:
        model, tokenizer, resolved = load_native_model(checkpoint, use_moe, device)
    except Exception as error:
        answer = f"模型加载失败：{error}"
        st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        try:
            for piece in generate_stream(
                model, tokenizer, st.session_state.messages, thinking=thinking,
                temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                max_context=max_context,
            ):
                answer += piece
                placeholder.markdown(display_text(answer, thinking) + "▌")
            answer = display_text(answer, thinking)
            placeholder.markdown(answer)
        except Exception as error:
            answer = f"生成失败：{error}"
            placeholder.error(answer)
        st.caption(f"checkpoint: {resolved} · thinking: {thinking}")
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    st.title("EvoMind Studio")
    st.caption("YH 的本地语言模型交互台")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    settings = sidebar()
    chat_tab, vision_tab = st.tabs(["💬 Text Chat", "🖼️ Vision"])
    with chat_tab:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        render_pending_response(*settings)
    with vision_tab:
        st.subheader("Dense VLM")
        st.info("视觉交互将在单图模型完成训练与验收后开放。")
        st.markdown("SigLIP2 → Projector → Dense LLM → 图文回答")
    # Root-level chat_input uses Streamlit's fixed-bottom composer.
    user_text = st.chat_input("输入问题，Enter 发送")
    if user_text:
        if not settings[0]:
            st.error("请先在侧边栏填写本地 checkpoint 路径。")
        else:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.pending_response = True
            st.rerun()


if __name__ == "__main__":
    main()
