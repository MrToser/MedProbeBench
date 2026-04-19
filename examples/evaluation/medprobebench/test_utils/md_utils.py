"""
Shared utilities for markdown processing.
"""

import re
from typing import Tuple, Dict, Optional


def extract_references_section(md_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract the References section, returning (remaining_content, references_text).
    
    Supported formats:
    - # References
    - # **References**
    - ## Reference
    - ### **REFERENCES:**
    - # 5. References
    - ## 10. Reference
    etc.
    """
    # Very permissive regex pattern, supports numeric prefixes
    pattern = re.compile(
        r'''
        ^                                   # Start of line
        (\#{1,6})                          # group 1: header level (1-6 #'s)
        \s*                                # Optional whitespace
        (?:\d+\.?)?                        # Optional numbering (e.g. "5." or "10")
        \s*                                # Optional whitespace
        (?:\*\*|__)?                       # Optional bold start (** or __)
        \s*                                # Optional whitespace
        references?                         # "reference" or "references"
        \s*                                # Optional whitespace
        (?:\*\*|__)?                       # Optional bold end
        \s*                                # Optional whitespace
        [:\-]?                             # Optional colon or hyphen
        \s*                                # Optional whitespace
        $                                  # End of line
        ''',
        re.MULTILINE | re.IGNORECASE | re.VERBOSE
    )
    
    match = pattern.search(md_text)
    if not match:
        return md_text, None
    
    # Extract References header and content
    ref_start = match.start()
    ref_header_level = len(match.group(1))
    ref_header = match.group(0).strip()
    
    # Find the end of the References content (next same-level or higher-level header)
    content_after_header = md_text[match.end():].lstrip('\n')
    
    # Search for the next same-level or higher-level header
    next_header_pattern = re.compile(
        rf'^(\#{{{1},{ref_header_level}}})\s+',
        re.MULTILINE
    )
    next_match = next_header_pattern.search(content_after_header)
    
    if next_match:
        ref_content = content_after_header[:next_match.start()].strip()
        content_without_refs = md_text[:ref_start] + content_after_header[next_match.start():]
    else:
        # References is the last section
        ref_content = content_after_header.strip()
        content_without_refs = md_text[:ref_start]
    
    references_section = f"{ref_header}\n\n{ref_content}"
    
    return content_without_refs.strip(), references_section

def extract_references_section(md_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract the References section, returning (remaining_content, references_text).

    Supported formats:
    - # References
    - ## **REFERENCES:**
    - # 5. References
    - References
    - **References**
    - 10. REFERENCES:
    """

    # ---------- Stage 1: ATX header (with #, strong signal) ----------
    pattern_atx = re.compile(
        r'''
        ^
        (\#{1,6})              # group 1: header level
        \s*
        (?:\d+\.?)?            # Optional numbering
        \s*
        (?:\*\*|__)?           # Optional bold start
        \s*
        references?
        \s*
        (?:\*\*|__)?           # Optional bold end
        \s*
        [:\-]?                 # Optional colon or hyphen
        \s*
        $
        ''',
        re.MULTILINE | re.IGNORECASE | re.VERBOSE
    )

    match = pattern_atx.search(md_text)

    # ---------- Stage 2: Plain header (no #, weak signal fallback) ----------
    pattern_plain = re.compile(
        r'''
        ^
        \s*
        (?:\d+\.?)?            # Optional numbering
        \s*
        (?:\*\*|__)?           # Optional bold
        \s*
        references?
        \s*
        (?:\*\*|__)?           # Optional bold
        \s*
        [:\-]?                 # Optional colon or hyphen
        \s*
        $
        ''',
        re.MULTILINE | re.IGNORECASE | re.VERBOSE
    )

    if match:
        ref_header_level = len(match.group(1))
    else:
        match = pattern_plain.search(md_text)
        if not match:
            return md_text, None
        ref_header_level = 0  # Indicates plain mode (no #)

    # ---------- Extract header ----------
    ref_start = match.start()
    ref_header = match.group(0).strip()

    content_after_header = md_text[match.end():].lstrip('\n')

    # ---------- Find the end of References ----------
    if ref_header_level > 0:
        # ATX: next same-level or higher-level header
        next_header_pattern = re.compile(
            rf'^(\#{{1,{ref_header_level}}})\s+',
            re.MULTILINE
        )
    else:
        # Plain: stop at any ATX header
        next_header_pattern = re.compile(
            r'^\#{1,6}\s+',
            re.MULTILINE
        )

    next_match = next_header_pattern.search(content_after_header)

    if next_match:
        ref_content = content_after_header[:next_match.start()].strip()
        content_without_refs = (
            md_text[:ref_start] +
            content_after_header[next_match.start():]
        )
    else:
        # References is the last section
        ref_content = content_after_header.strip()
        content_without_refs = md_text[:ref_start]

    references_section = f"{ref_header}\n\n{ref_content}"

    return content_without_refs.strip(), references_section

def parse_references(ref_section: str) -> Dict[str, str]:
    """
    Parse references text, returning a {number: citation_text} dictionary.
    
    Supported formats:
    - 1. Author et al. Title...
    - [1] Author et al. Title...
    - (1) Author et al. Title...
    - 1) Author et al. Title...
    - 1、2、3. Author et al. Title... (Chinese enumeration comma separated)
    - [1、2] Author et al. Title...
    """
    references = {}
    
    # Remove the header line
    lines = ref_section.split('\n')
    content_lines = []
    started = False
    
    for line in lines:
        # Skip header lines
        if not started and re.match(r'^\#{1,6}\s*', line):
            continue
        started = True
        content_lines.append(line)
    
    ref_text = '\n'.join(content_lines).strip()
    if not ref_text:
        return references
    
    # Method 1: Parse line by line, supports enumeration comma separation
    lines = ref_text.split('\n')
    current_nums = []  # Supports multiple numbers pointing to the same citation
    current_text = []
    
    for line in lines:
        # Match citation numbers (supports multiple formats, including enumeration commas)
        # Matches: [1、2、3] or 1、2、3. or [1] etc.
        num_match = re.match(
            r'^\s*[\[\(]?([\d、,\s]+)[\]\)\.:\-\s]+(.*)$',
            line.strip()
        )
        
        if num_match:
            # Save the previous citation
            if current_nums and current_text:
                combined_text = ' '.join(current_text).strip()
                for num in current_nums:
                    references[num] = combined_text
            
            # Extract all numbers (supports 、 , and space separators)
            num_str = num_match.group(1)
            # Split numbers: supports 、 , and space
            nums = re.split(r'[、,\s]+', num_str.strip())
            current_nums = [n.strip() for n in nums if n.strip().isdigit()]
            
            text_part = num_match.group(2).strip()
            current_text = [text_part] if text_part else []
        elif current_nums and line.strip():
            # Continue current citation (multi-line case)
            current_text.append(line.strip())
    
    # Save the last citation
    if current_nums and current_text:
        combined_text = ' '.join(current_text).strip()
        for num in current_nums:
            references[num] = combined_text
    
    # Method 2: If method 1 failed, try single-line format
    if not references:
        ref_pattern = r'[\[\(]?([\d、,\s]+)[\]\)\.:\-\s]+(.+?)(?=\n\s*[\[\(]?[\d、,\s]+[\]\)\.:\-\s]|$)'
        matches = re.findall(ref_pattern, ref_text, re.DOTALL)
        for num_str, ref_text_content in matches:
            nums = re.split(r'[、,\s]+', num_str.strip())
            nums = [n.strip() for n in nums if n.strip().isdigit()]
            combined_text = ref_text_content.strip().replace('\n', ' ')
            for num in nums:
                references[num] = combined_text
    
    return references
