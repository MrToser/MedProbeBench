import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict


SECTION_GROUPS = [
    {"definition"},
    {"icd-o coding", "icd-11 coding", "icd coding", "icd-o", "icd-11"},
    {"related terminology", "terminology"},
    {"subtype", "subtypes", "subtype(s)"},
    {"localization", "location"},
    {"clinical features", "clinical"},
    {"imaging"},
    {"spread"},  # optional
    {"epidemiology"},
    {"etiology", "aetiology"},
    {"pathogenesis"},
    {"macroscopic appearance", "macroscopic", "gross appearance"},
    {"histopathology", "histology"},
    {"immunophenotype", "immunohistochemistry", "ihc"},
    {"differential diagnosis"},  # optional
    {"cytology"},
    {"diagnostic molecular pathology", "molecular pathology", "molecular"},
    {"essential and desirable diagnostic criteria", "diagnostic criteria"},
    {"grading", "staging", "grading / staging"},
    {"prognosis and prediction", "prognosis", "prediction"},
]

OPTIONAL_SECTIONS = {"spread", "differential diagnosis"}
MATCH_THRESHOLD = 0.7


def extract_headers(content: str, max_level: int = 2) -> Tuple[List[Dict], List[str]]:
    """
    Extract headers, returning a structured header list and a list of header texts for matching.
    
    Args:
        content: markdown content
        max_level: maximum header level (1=only #, 2=# and ##, etc.)
    
    Returns:
        headers_info: [{"level": 1, "title": "xxx", "raw": "# xxx"}, ...]
        headers_for_match: ["definition", "clinical features", ...] for section matching
    """
    all_headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)

    headers_info = []
    headers_for_match = []

    for level_str, title in all_headers:
        level = len(level_str)
        clean_title = re.sub(r'^\d+\.\s*', '', title.strip())  # Remove numbering
        
        headers_info.append({
            "level": level,
            "title": clean_title,
            "raw": f"{level_str} {title}",
        })
        
        # Only headers with level <= max_level participate in section matching
        if level <= max_level:
            headers_for_match.append(clean_title.lower())

    return headers_info, headers_for_match


def check_sections_exist(headers: List[str], threshold: float):
    """Check whether required sections exist."""
    matched = []
    missing = []

    for group in SECTION_GROUPS:
        is_optional = bool(group.intersection(OPTIONAL_SECTIONS))
        found = False

        for header in headers:
            if any(keyword in header for keyword in group):
                matched.append(header)
                found = True
                break

        if not found and not is_optional:
            missing.append(list(group)[0])

    match_ratio = len(matched) / len(SECTION_GROUPS)
    return match_ratio >= threshold, match_ratio, matched, missing


def analyze_header_structure(headers_info: List[Dict]) -> Dict:
    """Analyze the header hierarchy structure."""
    level_counts = {}
    for h in headers_info:
        level = h["level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    return {
        "total_headers": len(headers_info),
        "level_counts": level_counts,
        "max_level": max(level_counts.keys()) if level_counts else 0,
        "has_h1": 1 in level_counts,
        "has_h2": 2 in level_counts,
    }


def check_format(filepath: Path, threshold: float, max_level: int = 2):
    """
    Check file format.
    
    Args:
        filepath: file path
        threshold: section match threshold
        max_level: maximum header level used for section matching
    """
    content = filepath.read_text(encoding="utf-8")
    # Extract headers first
    headers_info, headers_for_match = extract_headers(content, max_level=max_level)
    # Analyze header structure
    structure = analyze_header_structure(headers_info)
    # Determine match
    sections_match, match_ratio, matched, missing = check_sections_exist(headers_for_match, threshold)

    # Format is valid if section match ratio meets the threshold
    is_valid = sections_match

    details = {
        "structure": structure,
        "headers_info": headers_info,
        "headers_for_match": headers_for_match,
        "sections_match": sections_match,
        "match_ratio": match_ratio,
        "matched_sections": matched,
        "missing_sections": missing,
    }

    return is_valid, details


def iter_md_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return sorted(input_path.glob("*.md"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")