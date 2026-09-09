"""Sync the generated report and its figures to the user-approved Obsidian vault.

The existing MLLM重点笔记.md is never overwritten by this script. Generated
report files carry an ownership header; edited/unowned targets are refused.
"""
from pathlib import Path
import hashlib
import os
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
VAULT = Path("D:/#obsidian/YH")
HEADER = "<!-- evomind-generated-report: source E:/project/Learning/evomind -->"


def main():
    if not VAULT.is_dir():
        raise FileNotFoundError("Approved Obsidian vault is not mounted; project report remains on E drive")
    source = ROOT / "EVOMIND_TECHNICAL_REPORT.md"
    target = VAULT / "Projects/evomind技术报告.md"
    if target.exists() and not target.read_text(encoding="utf-8").startswith(HEADER + "\n"):
        raise FileExistsError("Refusing to overwrite an unowned Obsidian note")
    content = source.read_text(encoding="utf-8")
    attachments = VAULT / "Attachments/evomind"

    def figure(match):
        label, link = match.groups()
        if re.match(r"[a-z]+://", link):
            return match.group(0)
        path = (ROOT / link).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Report figure is missing or outside this project: {link}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = attachments / f"{digest[:12]}_{path.name}"
        attachments.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError("Content-addressed Obsidian attachment was modified")
        else:
            shutil.copy2(path, destination)
        return f"![{label}](../Attachments/evomind/{destination.name})"

    content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", figure, content)
    content = HEADER + "\n" + content + "\n\n完整命令与原始JSONL见 `E:/project/Learning/evomind`。本页为自动生成副本，学习批注请写入 [[MLLM重点笔记]]。\n"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)
    print(f"Obsidian report synchronized: {target}")


if __name__ == "__main__":
    main()
