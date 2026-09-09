"""Bounded local tool validation; never evaluates arbitrary Python or uses SIGALRM.

Only arithmetic is implemented here. Callers retain their original mock tables
and return values: these tools are NOT live weather, exchange-rate or translation
services, and the Agent reward ground truth must not be inferred from this file.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re

MAX_EXPRESSION_CHARS = 512
MAX_AST_NODES = 128
MAX_AST_DEPTH = 16
MAX_INTEGER_BITS = 256
MAX_EXPONENT = 64
MAX_ARGUMENT_CHARS = 8192
MAX_TOOL_CALLS = 8
MAX_TOOL_TEXT_CHARS = 65536


class ToolInputError(ValueError):
    """Invalid or resource-exceeding tool input (safe to expose to the model)."""


def _number(value):
    if type(value) not in (int, float):
        raise ToolInputError("Only real numbers are allowed")
    if isinstance(value, int):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise ToolInputError("Integer exceeds 256 bits")
    elif not math.isfinite(value) or abs(value) > 1e100:
        raise ToolInputError("Non-finite or excessive numeric magnitude")
    return value


def safe_arithmetic(expression):
    """Evaluate a tiny arithmetic grammar with bounded parse and operation cost.

    Names/attributes/containers/subscripts/comprehensions are forbidden. The only
    calls are sqrt(x), abs(x), floor(x), ceil(x), with one positional argument.
    Normalized ^ means exponentiation, as in the upstream tool. No eval/exec,
    Python compilation, subprocess, signal, file, network, or arbitrary callback.
    """
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_EXPRESSION_CHARS:
        raise ToolInputError("Expression must contain 1..512 characters")
    expression = expression.translate(str.maketrans({"×": "*", "÷": "/", "−": "-", "（": "(", "）": ")"}))
    expression = expression.replace("^", "**").replace("²", "**2").replace("³", "**3").strip()
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolInputError("Normalized expression exceeds 512 characters")
    try:
        root = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ToolInputError("Invalid arithmetic syntax") from exc
    stack, nodes = [(root, 0)], 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            raise ToolInputError("Expression exceeds AST node/depth limit")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    functions = {"sqrt": math.sqrt, "abs": abs, "floor": math.floor, "ceil": math.ceil}

    def visit(node):
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand)
            return _number(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise ToolInputError("Exponent exceeds absolute limit 64")
                # Preflight before materializing a potentially enormous integer.
                if type(left) is int and type(right) is int and right > 0 and abs(left) > 1:
                    if (abs(left).bit_length() - 1) * right >= MAX_INTEGER_BITS:
                        raise ToolInputError("Power exceeds integer bit limit")
                return _number(operator.pow(left, right))
            fn = binary.get(type(node.op))
            if fn is not None:
                return _number(fn(left, right))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            return _number(functions[node.func.id](visit(node.args[0])))
        raise ToolInputError("Unsupported arithmetic syntax")

    try:
        return visit(root.body)
    except (ArithmeticError, ValueError) as exc:
        if isinstance(exc, ToolInputError):
            raise
        raise ToolInputError("Arithmetic domain or division error") from exc


def _reject_constant(value):
    raise ToolInputError("Non-finite JSON constants are forbidden")


def parse_arguments(raw):
    if isinstance(raw, str):
        if len(raw) > MAX_ARGUMENT_CHARS:
            raise ToolInputError("Arguments exceed 8192 characters")
        try:
            raw = json.loads(raw, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise ToolInputError("Arguments must be a JSON object") from exc
    if not isinstance(raw, dict) or len(raw) > 16:
        raise ToolInputError("Arguments must be an object with at most 16 fields")
    for key, value in raw.items():
        if not isinstance(key, str) or len(key) > 64:
            raise ToolInputError("Invalid argument name")
        if isinstance(value, str):
            if len(value) > MAX_ARGUMENT_CHARS:
                raise ToolInputError("Argument string is too long")
        elif type(value) in (int, float):
            _number(value)
        else:
            raise ToolInputError("Only scalar string/number arguments are supported")
    if sum(len(value) for value in raw.values() if isinstance(value, str)) > MAX_ARGUMENT_CHARS:
        raise ToolInputError("Combined string arguments are too long")
    return raw


_FIELDS = {
    "calculate_math": ({"expression": str}, set()),
    "unit_converter": ({"value": float, "from_unit": str, "to_unit": str}, set()),
    "get_current_weather": ({"location": str, "unit": str}, {"unit"}),
    "get_current_time": ({"timezone": str}, {"timezone"}),
    "get_exchange_rate": ({"from_currency": str, "to_currency": str}, set()),
    "translate_text": ({"text": str, "target_language": str}, set()),
    "random_number": ({"min": int, "max": int}, {"min", "max"}),
    "text_length": ({"text": str}, set()),
}


def validate_tool_arguments(name, raw):
    if not isinstance(name, str) or name not in _FIELDS:
        raise ToolInputError("Unknown tool")
    args = parse_arguments(raw)
    fields, optional = _FIELDS[name]
    if set(args) - set(fields) or set(fields) - optional - set(args):
        raise ToolInputError("Missing or unexpected tool arguments")
    for key, value in args.items():
        expected = fields[key]
        if expected is str:
            if not isinstance(value, str) or not value.strip():
                raise ToolInputError(f"{key} must be a nonempty string")
        elif expected is int:
            if type(value) is not int:
                raise ToolInputError(f"{key} must be an integer")
        elif type(value) not in (int, float):
            raise ToolInputError(f"{key} must be numeric")
    if name == "random_number" and args.get("min", 0) > args.get("max", 100):
        raise ToolInputError("min must not exceed max")
    return args


def parse_tool_calls(text):
    """Parse at most eight structured calls; malformed JSON never escapes."""
    if not isinstance(text, str) or len(text) > MAX_TOOL_TEXT_CHARS:
        raise ToolInputError("Tool-call text exceeds 65536 characters")
    matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL))
    if len(matches) > MAX_TOOL_CALLS:
        raise ToolInputError("More than eight tool calls in one turn")
    calls = []
    for match in matches:
        if len(match.group(1)) > MAX_ARGUMENT_CHARS:
            continue
        try:
            call = json.loads(match.group(1).strip(), parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            continue
        if (isinstance(call, dict) and isinstance(call.get("name"), str)
                and 0 < len(call["name"]) <= 64 and isinstance(call.get("arguments", {}), (dict, str))):
            calls.append(call)
    return calls
