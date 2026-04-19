"""
Standardize markdown section headers to canonical names.

Approach:
1. Extract all levels of sections and their content
2. Match section titles to CANONICAL_SECTIONS using similarity
3. Regenerate a standard MD file with only level-1 headers (#)
4. Merge sub-header content into the corresponding parent section
"""

import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# Import shared utilities
from md_utils import extract_references_section

# 20 canonical section names
CANONICAL_SECTIONS = [
    "Definition",
    "ICD-O coding / ICD-11 coding",
    "Related terminology",
    "Subtype(s)",
    "Localization",
    "Clinical features",
    "Imaging",
    "Spread",
    "Epidemiology",
    "Etiology",
    "Pathogenesis",
    "Macroscopic appearance",
    "Histopathology",
    "Immunophenotype",
    "Differential diagnosis",
    "Cytology",
    "Diagnostic molecular pathology",
    "Essential and desirable diagnostic criteria",
    "Grading / Staging",
    "Prognosis and prediction",
]

# Keyword mapping (for fast matching)
KEYWORD_MAPPING = {
    "definition": "Definition",
    "icd": "ICD-O coding / ICD-11 coding",
    "terminology": "Related terminology",
    "synonym": "Related terminology",
    "subtype": "Subtype(s)",
    "variant": "Subtype(s)",
    "localization": "Localization",
    "location": "Localization",
    "site": "Localization",
    "clinical": "Clinical features",
    "symptom": "Clinical features",
    "imaging": "Imaging",
    "radiolog": "Imaging",
    "spread": "Spread",
    "metastas": "Spread",
    "epidemiology": "Epidemiology",
    "incidence": "Epidemiology",
    "etiology": "Etiology",
    "aetiology": "Etiology",
    "risk factor": "Etiology",
    "pathogenesis": "Pathogenesis",
    "macroscopic": "Macroscopic appearance",
    "gross": "Macroscopic appearance",
    "histopathology": "Histopathology",
    "histology": "Histopathology",
    "microscop": "Histopathology",
    "immunophenotype": "Immunophenotype",
    "immunohistochem": "Immunophenotype",
    "ihc": "Immunophenotype",
    "differential": "Differential diagnosis",
    "cytology": "Cytology",
    "molecular": "Diagnostic molecular pathology",
    "genetic": "Diagnostic molecular pathology",
    "diagnostic criteria": "Essential and desirable diagnostic criteria",
    "criteria": "Essential and desirable diagnostic criteria",
    "grading": "Grading / Staging",
    "staging": "Grading / Staging",
    "grade": "Grading / Staging",
    "prognosis": "Prognosis and prediction",
    "prediction": "Prognosis and prediction",
    "survival": "Prognosis and prediction",
    "outcome": "Prognosis and prediction",
}

SIMILARITY_THRESHOLD = 0.5


def extract_sections_all_levels(content: str) -> List[Dict]:
    """
    Extract all levels of sections and their content.
    Returns a flat list where each element contains level, title, content.
    """
    lines = content.split('\n')
    sections = []
    current = None
    content_lines = []
    
    # Match all header levels (1-6)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save the previous section
            if current is not None:
                current["content"] = '\n'.join(content_lines).strip()
                sections.append(current)
            
            # Clean title: remove numbering
            title = re.sub(r'^\d+\.\s*', '', match.group(2).strip())
            current = {"level": len(match.group(1)), "title": title}
            content_lines = []
        else:
            content_lines.append(line)
    
    # Save the last section
    if current is not None:
        current["content"] = '\n'.join(content_lines).strip()
        sections.append(current)
    
    return sections


def match_to_canonical(title: str) -> Optional[str]:
    """Match a title to a canonical section name."""
    title_lower = title.lower()
    
    # 1. Keyword matching (priority)
    for keyword, canonical in KEYWORD_MAPPING.items():
        if keyword in title_lower:
            return canonical
    
    # 2. Similarity matching
    best_match, best_score = None, 0
    for canonical in CANONICAL_SECTIONS:
        score = SequenceMatcher(None, title_lower, canonical.lower()).ratio()
        if score > best_score:
            best_match, best_score = canonical, score
    
    return best_match if best_score >= SIMILARITY_THRESHOLD else None


def standardize_markdown(content: str) -> Tuple[str, Dict]:
    """
    Standardize markdown:
    1. Match all sections to canonical names
    2. Merge content of sections mapped to the same canonical name
    3. Output with only level-1 headers (#), preserving the References section
    
    Returns: (new_content, stats)
    """
    # Extract the References section (using shared function)
    content, references_section = extract_references_section(content)
    
    sections = extract_sections_all_levels(content)
    
    # Match and collect content (merge content for the same canonical name)
    canonical_contents: Dict[str, List[str]] = {s: [] for s in CANONICAL_SECTIONS}
    matched, unmatched = [], []
    
    # Current parent section (for handling sub-headers)
    current_parent_canonical = None
    
    for sec in sections:
        title = sec["title"]
        level = sec["level"]
        section_content = sec["content"]
        
        # Try to match the current title
        canonical = match_to_canonical(title)
        
        if canonical:
            # Match succeeded, update current parent
            current_parent_canonical = canonical
            if section_content:
                canonical_contents[canonical].append(section_content)
                matched.append((title, canonical, level))
        elif current_parent_canonical and level > 1:
            # Sub-header not matched to a canonical section, assign to current parent
            # Preserve sub-header text as part of the content (converted to bold)
            if section_content:
                sub_content = f"**{title}**\n\n{section_content}"
                canonical_contents[current_parent_canonical].append(sub_content)
                matched.append((title, f"{current_parent_canonical} (sub)", level))
        elif section_content:
            # Level-1 header not matched, record as unmatched
            unmatched.append(title)
    
    # Generate standardized MD (using only level-1 headers)
    output_lines = []
    for canonical in CANONICAL_SECTIONS:
        contents = canonical_contents[canonical]
        if contents:
            output_lines.append(f"# {canonical}")
            output_lines.append("")
            # Merge all content, separated by double newlines
            output_lines.append("\n\n".join(contents))
            output_lines.append("")
    
    # Append the References section
    if references_section:
        output_lines.append(references_section)
    
    stats = {
        "matched": matched,
        "unmatched": unmatched,
        "sections_with_content": sum(1 for c in canonical_contents.values() if c),
        "references_included": references_section is not None,
    }
    return '\n'.join(output_lines), stats


def standardize_file(input_path: Path, output_path: Path, verbose: bool = False) -> Dict:
    """
    Standardize a single file.
    
    Returns:
        stats dict containing matched, unmatched, sections_with_content
    """
    content = input_path.read_text(encoding='utf-8')
    standardized, stats = standardize_markdown(content)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(standardized, encoding='utf-8')
    
    if verbose:
        print(f"\n📄 {input_path.name}:")
        for item in stats["matched"]:
            if len(item) == 3:
                orig, canon, level = item
                level_indicator = "#" * level
                print(f"   ✓ [{level_indicator}] '{orig}' → '{canon}'")
            else:
                orig, canon = item
                print(f"   ✓ '{orig}' → '{canon}'")
        for orig in stats["unmatched"]:
            print(f"   ✗ '{orig}' (unmatched)")
        print(f"   📊 Sections with content: {stats['sections_with_content']}/{len(CANONICAL_SECTIONS)}")
    
    return stats