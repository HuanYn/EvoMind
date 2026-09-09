import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import re
import json
import time
import random
import argparse
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from trainer.evomind_tools import (safe_arithmetic, parse_tool_calls, validate_tool_arguments,
                                  ToolInputError, MAX_TOOL_CALLS, MAX_TOOL_TEXT_CHARS, MAX_ARGUMENT_CHARS)
warnings.filterwarnings('ignore')

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_arithmetic(args["expression"]))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_artifacts(args):
    """Resolve every native weight before importing/constructing any model."""
    root = Path(__file__).resolve().parents[1]
    if args.model_format == 'native':
        if args.hidden_size != 768 or args.num_hidden_layers != 8:
            raise ValueError('The shared strict native loader requires the official 768-wide, 8-layer model')
        tokenizer = Path(args.load_from).resolve()
        if tokenizer != root / 'model':
            raise ValueError('Native evaluation uses the pinned project/model tokenizer; --load_from must select it')
        suffix = '_moe' if args.use_moe else ''
        checkpoint = Path(args.save_dir).resolve() / f'{args.weight}_{args.hidden_size}{suffix}.pth'
        paths = {'checkpoint': checkpoint, 'tokenizer': tokenizer / 'tokenizer.json',
                 'tokenizer_config': tokenizer / 'tokenizer_config.json'}
        if args.lora:
            paths['lora'] = Path(args.lora).resolve()
    else:
        if args.lora:
            raise ValueError('--lora is supported only with the strict native loader')
        directory = Path(args.load_from).resolve()
        weights = sorted([*directory.glob('*.safetensors'), *directory.glob('pytorch_model*.bin')])
        if not weights:
            raise FileNotFoundError(f'No local HF model weights: {directory}')
        paths = {'config': directory / 'config.json',
                 **{f'hf/{p.name}': p for p in [*weights, *directory.glob('*.json')]}}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/empty required evaluation artifact: {path}')
    return paths


def evaluation_provenance(args):
    root = Path(__file__).resolve().parents[1]
    paths = {name: root / name for name in ('scripts/eval_toolcall.py', 'trainer/evomind_tools.py',
                                           'scripts/evomind_load_text_model.py', 'model/model_minimind.py',
                                           'model/model_lora.py')}
    if args.backend == 'local':
        paths.update(model_artifacts(args))
    return {name: {'path': str(path.resolve()), 'sha256': file_sha256(path)} for name, path in paths.items()}


def init_model(args):
    artifacts = model_artifacts(args)
    import datasets  # Windows DLL initialization order; local backend only.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trainer.trainer_utils import get_model_params
    if args.model_format == 'native':
        from evomind_load_text_model import load_model
        model, tokenizer = load_model(artifacts['checkpoint'], artifacts.get('lora'),
                                      device=args.device, use_moe=bool(args.use_moe))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=False, local_files_only=True)
    get_model_params(model, model.config)
    return model.to(dtype=torch.float32 if torch.device(args.device).type == 'cpu' else torch.float16).eval().requires_grad_(False).to(args.device), tokenizer


def parse_tool_call_from_text(content):
    tool_calls = []
    for i, data in enumerate(parse_tool_calls(content)):
        raw = data.get("arguments", {})
        tool_calls.append({"id": f"call_{i}", "function": {"name": data["name"],
            "arguments": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)}})
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    if not isinstance(name, str):
        return {"error": "Invalid tool name"}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        return fn(validate_tool_arguments(name, raw_args))
    except (ToolInputError, ArithmeticError, TypeError, ValueError) as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=args.max_new_tokens, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    started = time.monotonic()
    for chunk_index, chunk in enumerate(response):
        if chunk_index >= 32768 or time.monotonic() - started > 180:
            response.close()
            raise ToolInputError('API stream exceeds chunk/time limit')
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
            if len(content) > MAX_TOOL_TEXT_CHARS:
                response.close()
                raise ToolInputError('API content exceeds character limit')
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                if not isinstance(idx, int) or not 0 <= idx < MAX_TOOL_CALLS:
                    raise ToolInputError("API tool call index exceeds limit")
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
                    if (len(tool_calls[idx]["function"]["arguments"]) > MAX_ARGUMENT_CHARS
                            or len(tool_calls[idx]["function"]["name"]) > 64):
                        response.close()
                        raise ToolInputError('API function arguments/name exceed limit')
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    if not 1 <= args.max_turns <= 16:
        raise ValueError("max_turns must be between 1 and 16")
    records = []
    allowed_names = {tool['function']['name'] for tool in tools}
    for turn in range(args.max_turns):
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            records.append({"assistant": content, "tool_results": []})
            return {"prompt": prompt, "status": "complete", "unfinished": False, "turns": records,
                    "tool_semantics": "upstream_mock_not_live_services", "model_quality_score": None}
        if len(tool_calls) > MAX_TOOL_CALLS:
            raise ToolInputError("More than eight tool calls in one turn")
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        record = {"assistant": content, "tool_results": []}
        records.append(record)
        for tc in tool_calls:
            name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = (execute_tool(tc if args.backend == 'local' else name, arguments)
                      if name in allowed_names else {"error": "Tool is not enabled for this case"})
            record["tool_results"].append({"name": name, "arguments": arguments, "result": result})
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})
    print('[ToolCall] max_turns reached; episode is unfinished (not a successful answer).')
    return {"prompt": prompt, "status": "max_turns", "unfinished": True, "turns": records,
            "tool_semantics": "upstream_mock_not_live_services", "model_quality_score": None}


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='../model', type=str, help="native 模式选择固定 tokenizer 目录；hf 模式选择本地完整模型目录")
    parser.add_argument('--model_format', default='native', choices=['native', 'hf'], help='Explicit format, never inferred from substrings in a path')
    parser.add_argument('--lora', type=Path, help='Strict native adapter checkpoint, applied on top of --weight')
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='auto', type=str, help="运行设备；auto 仅在本地后端检测 CUDA")
    parser.add_argument('--max_turns', default=3, type=int, help="每例最多生成轮数，1..16；超限标记 unfinished")
    parser.add_argument('--auto', action='store_true', help="非交互执行固定测试集")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_json', type=Path, help="新建结构化诊断结果，不报告虚构质量分数")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    args = parser.parse_args()
    if not 1 <= args.max_turns <= 16 or not 1 <= args.max_new_tokens <= 8192:
        parser.error('max_turns must be 1..16 and max_new_tokens 1..8192')
    if args.output_json and args.output_json.exists():
        raise FileExistsError(args.output_json)
    print('[ToolCall] Upstream mock tools retained: weather/rates/translations are not live facts.')
    provenance = evaluation_provenance(args)

    model = tokenizer = client = None
    if args.backend == 'local':
        import datasets
        import torch
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, tokenizer = init_model(args)
    else:
        from openai import OpenAI  # Optional dependency: local evaluation never imports it.
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url, timeout=60.0, max_retries=0)

    input_mode = 0 if args.auto else int(input('[0] 自动测试\n[1] 手动输入\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    results = []
    for index, case in enumerate(cases):
        if not case["prompt"]: break
        random.seed(args.seed + index)
        if args.backend == 'local':
            torch.random.default_generator.manual_seed(args.seed + index)
            if torch.device(args.device).type == 'cuda':
                with torch.cuda.device(args.device):
                    torch.cuda.manual_seed(args.seed + index)
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        results.append(run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client))
        print('\n' + '-' * 50 + '\n')
    if args.output_json:
        if evaluation_provenance(args) != provenance:
            raise RuntimeError('Evaluation source/model artifact changed during execution; refusing completion receipt')
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(args.output_json.name + '.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump({"status": "complete", "scope": "bounded_mock_toolcall_diagnostic_not_accuracy",
                       "backend": args.backend, "seed": args.seed, "max_turns": args.max_turns,
                       "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
                       "inference_dtype": ('float32' if torch.device(args.device).type == 'cpu' else 'float16') if args.backend == 'local' else 'remote_unverified',
                       "api_model": args.api_model if args.backend == 'api' else None,
                       "provenance": provenance, "cases": results}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(args.output_json)


if __name__ == "__main__":
    main()
