"""
Convert extracted_claims to standardized JSONL format

Description:
--------------------
Convert JSON files from extracted_claims directory to standardized JSONL format.
Each JSONL record contains: medical guideline info, system prompt, sections list,
original content, extracted claims, and references.

Input:
------------
- extracted_claims/*.json : Extracted claims JSON files
- md-manual-check-add-box/*.md : Original medical guideline markdown files

Output:
-------------
- converted_claims.jsonl : Standardized JSONL format file

Output Format Example:
----------------------------------
{
  "id": "tumor_name_id",
  "System": {"role_and_constraints": "You are a medical guideline writer..."},
  "sections": ["Definition", "ICD-O coding / ICD-11 coding", ...],
  "prompt": "Tumor Name",
  "content": "# Original Markdown Content...",
  "claims": [{"id": "C001", "claim": "...", "reference": ["1"], ...}],
  "references": {"1": "Reference text..."}
}

Usage:
----------------
python scripts/convert_to_jsonl_format.py

Notes:
----------------
- Automatically reads original reports from md-manual-check-add-box
- Automatically converts the reference field from string "[1]" to array ["1"]
- Automatically renames the claim's content field to claim
- Output file is overwritten on each run
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

# Standard sections list (from SECTION_Template_EN.md)
STANDARD_SECTIONS = [
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
    "Prognosis and prediction"
]

# System prompt for medical guideline generation
SYSTEM_PROMPT = "You are a medical guideline writer. Based on authoritative literature, write a chapter of an authoritative medical report. You need to follow the chapter format below."

def create_guideline_id(guideline_name):
    """Convert guideline name to ID format"""
    # Remove special characters and convert to lowercase with underscores
    id_str = guideline_name.lower()
    id_str = re.sub(r'[^\w\s-]', '', id_str)
    id_str = re.sub(r'[\s-]+', '_', id_str)
    return id_str

def parse_reference_string(ref_str):
    """Parse reference string like '[1]' or '[1, 2]' to list of numbers"""
    if not ref_str or ref_str.strip() == "":
        return []

    # Extract numbers from reference string
    numbers = re.findall(r'\d+', ref_str)
    return numbers

def read_markdown_content(md_path):
    """Read the original markdown file content"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def convert_claim_format(claim):
    """Convert claim from old format to new format"""
    new_claim = {
        "id": claim.get("id", ""),
        "claim": claim.get("content", ""),  # Change 'content' to 'claim'
        "reference": parse_reference_string(claim.get("reference", "")),  # Convert to array
        "type_knowledge": claim.get("type_knowledge", ""),
        "section": claim.get("section", "")
    }
    return new_claim

def convert_references_format(references):
    """
    Convert references from enriched format to output format.
    
    Input formats supported:
    1. Old format: {"1": "Reference text..."}
    2. Enriched format: {"1": {"text": "...", "pmid": "...", "urls": {...}}}
    
    Output format:
    {
        "1": {
            "text": "Reference text...",
            "pmid": "12345678",
            "doi": "10.xxxx/xxx",
            "urls": {
                "pubmed": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "doi": "https://doi.org/10.xxxx/xxx",
                "other": "https://...",
                "publisher": "https://..."
            }
        }
    }
    """
    new_refs = {}
    for key, value in references.items():
        str_key = str(key)
        # print("value:", value)
        if isinstance(value, str):
            # Old format: just text string
            new_refs[str_key] = {
                "text": value,
                "pmid": None,
                "doi": None,
                "urls": {
                    "pubmed": None,
                    "doi": None,
                    "other": None,
                    "publisher": None
                }
            }
        elif isinstance(value, dict):
            # Enriched format: already a dict with metadata
            new_refs[str_key] = {
                "text": value.get("text", ""),
                "pmid": value.get("pmid"),
                "doi": value.get("doi"),
                "urls": value.get("urls", {
                    "pubmed": None,
                    "doi": None,
                    "other": None,
                    "publisher": None
                })
            }
        else:
            # Fallback
            new_refs[str_key] = {
                "text": str(value),
                "pmid": None,
                "doi": None,
                "urls": {"pubmed": None, "doi": None, "other": None, "publisher": None}
            }
    
    return new_refs

def convert_json_to_jsonl(json_path, md_path, output_path):
    """Convert a single JSON file to JSONL format"""
    try:
        # Read extracted claims JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        guideline_name = data.get("guideline", "")

        # Read markdown content
        md_content = read_markdown_content(md_path)

        # Get references - support both 'reference' and 'references' keys
        raw_references = data.get("references", data.get("reference", {}))

        # Build output structure
        output = {
            "id": create_guideline_id(guideline_name),
            "System": {
                "role_and_constraints": SYSTEM_PROMPT
            },
            "sections": STANDARD_SECTIONS,
            "prompt": guideline_name,
            "content": md_content if md_content else f"# {guideline_name}\n\n(Content not available)",
            "claims": [convert_claim_format(claim) for claim in data.get("claims", [])],
            "references": convert_references_format(raw_references)
        }

        # Write to JSONL file (one JSON object per line)
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

        return True

    except Exception as e:
        print(f"Error converting {json_path.name}: {str(e)}")
        return False