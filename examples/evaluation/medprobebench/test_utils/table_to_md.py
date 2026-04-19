import json
import argparse
from pathlib import Path


def entry_to_md(entry: dict, output_dir: Path):
    tumor_id = entry.get("id", "unknown_id")
    sections = entry.get("sections", {})
    references = entry.get("references", {})

    md_lines = []

    # Sections (all level-1 headings, no main title to keep consistent)
    for section_title, content in sections.items():
        md_lines.append(f"# {section_title}\n")
        if isinstance(content, str) and content.strip():
            md_lines.append(content.strip() + "\n")
        else:
            md_lines.append("_Not specified._\n")

    # References → numbered list (1. 2. 3.)
    if references:
        md_lines.append("# References\n")
        for idx, (_, url) in enumerate(references.items(), start=1):
            md_lines.append(f"{idx}. {url}")

    output_path = output_dir / f"{tumor_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    return output_path


def jsonl_to_md(jsonl_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry_to_md(entry, output_dir)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] JSON decode error: {e}")

    print(f"Markdown files saved to: {output_dir.resolve()}")