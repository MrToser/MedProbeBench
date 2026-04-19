# -*- coding: utf-8 -*-
"""
Convert unstructured medical reports into structured table format by Section.

Input: Medical report + Table template (20 Sections)
Output: Progressively filled table (JSONL format)

Workflow:
1. Initialize empty table (20 Sections)
2. Iterate through each Section, call LLM to extract corresponding content
3. Fill extracted content into the table
4. Finally extract reference list
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field

from openai import OpenAI


# ============================================================================
# 20 Standard Sections with Full Descriptions (from SECTION_Template_EN.md)
# ============================================================================

SECTIONS = [
    ("Definition", 
     "Describe what the tumor is, its nature, which major/subclass CNS tumor category it belongs to, "
     "as well as its core pathological information, molecular pathological information, diagnostic features, "
     "and grading attributes."),
    
    ("ICD-O coding / ICD-11 coding", 
     "Provide the official codes for this tumor in the International Classification of Diseases system. "
     "(Note: In original files, these may appear as two separate sections: ICD-O coding and ICD-11 coding)"),
    
    ("Related terminology", 
     "Address historically or literature-used names that are no longer recommended, "
     "and clarify easily confused concepts."),
    
    ("Subtype(s)", 
     "Indicate whether the tumor has further subtypes or grading, or exists on a continuum "
     "(including NOS/NEC trigger statements that can be placed here)."),
    
    ("Localization", 
     "Describe the anatomical distribution characteristics and data of this tumor "
     "(common and possible sites of occurrence), and variations in growth and recurrence patterns "
     "at different locations."),
    
    ("Clinical features", 
     "Describe the common clinical manifestations (symptoms and signs) of patients. "
     "Explain what these symptoms indicate, location distribution, etc."),
    
    ("Imaging", 
     "Describe the specific or general manifestations of this tumor on imaging examinations "
     "such as CT, MRI, PET."),
    
    ("Spread", 
     "Describe the dissemination patterns and intracranial/extracranial metastatic manifestations "
     "of the tumor. (Note: Not all tumor types have this section)"),
    
    ("Epidemiology", 
     "Provide statistical indicators such as incidence and prevalence, susceptible populations, "
     "and other epidemiological characteristics related to disease occurrence. "
     "Subsections may include: 1) Incidence 2) Age and sex distribution"),
    
    ("Etiology", 
     "Describe germline, GWAS/SNP-related characteristics; relevant special tumor genetic susceptibility "
     "syndromes and their features; discuss currently known or unknown pathogenic factors, "
     "immune/metabolic-related factors, animal models, etc. "
     "Subsections may include: 1) Genetic susceptibility/factors 2) Risk factors 3) Other etiological factors"),
    
    ("Pathogenesis", 
     "Molecular mechanisms and pathophysiological processes of tumor development and progression. "
     "Subsections may include: 1) Cell of origin 2) Genetic profile/Genetics 3) Epigenetic changes"),
    
    ("Macroscopic appearance", 
     "Describe the morphology, boundaries, texture, characteristics and other anatomical properties "
     "of this tumor, and correlations with subtypes/grades."),
    
    ("Histopathology", 
     "Histopathological features including: 1) Cellular composition - microscopic findings including "
     "cell morphology, density, types, distribution 2) Mineralization and other degenerative features "
     "3) Vasculature - vascular distribution and morphology 4) Growth pattern 5) Proliferation markers (e.g., ki67)"),
    
    ("Immunophenotype", 
     "Immunohistochemical markers and their expression patterns relevant to diagnosis, grading, "
     "and differential diagnosis of this tumor."),
    
    ("Differential diagnosis", 
     "Diseases (tumor subtypes or non-neoplastic diseases, NEC/NOS) that need to be differentiated "
     "from this tumor in terms of imaging, symptoms, pathology, including similarities, distinguishing features, "
     "and differential methods. (Commonly uses format: 'reference entity → key differences → key tests') "
     "(Note: Not all tumor types have this section)"),
    
    ("Cytology", 
     "Observations and characteristics from cytological smear examination, "
     "and their role in diagnosis or differential diagnosis."),
    
    ("Diagnostic molecular pathology", 
     "Molecular pathological features and definitions related to diagnosis, differential diagnosis, "
     "and grading of this tumor; recommended/acceptable testing methods and method substitutability/"
     "non-substitutability, and methodologies."),
    
    ("Essential and desirable diagnostic criteria", 
     "List the essential conditions and ideal conditions, supportive conditions, NEC/NOS rules, "
     "and other conditions that must be met for diagnosing this tumor."),
    
    ("Grading / Staging", 
     "Grading or staging and criteria related to this subtype. "
     "(Note: In original files, Grading and Staging may be separate sections or combined as Staging)"),
    
    ("Prognosis and prediction", 
     "Prognostic (survival and recurrence) and predictive factors. Subsections may include: "
     "1) Clinical factors 2) Imaging 3) Surgery 4) Histological features "
     "5) CNS WHO grading 6) Genetic alterations 7) Treatment"),
]


# ============================================================================
# Prompt Templates (English) - Improved for strict section boundaries
# ============================================================================

SECTION_EXTRACT_PROMPT = """You are a medical literature expert. Please extract content ONLY related to "{section_name}" from the following medical report.

## Section Description
{section_name}: {section_desc}

## All 20 Sections (for reference - DO NOT include content belonging to other sections)
1. Definition - tumor definition, nature, classification, core pathological info
2. ICD-O coding / ICD-11 coding - ICD codes ONLY
3. Related terminology - historical names, synonyms
4. Subtype(s) - subtypes, variants
5. Localization - anatomical distribution, sites
6. Clinical features - symptoms and signs
7. Imaging - CT, MRI, PET findings
8. Spread - dissemination, metastasis patterns
9. Epidemiology - incidence, prevalence, demographics
10. Etiology - causes, risk factors, genetic susceptibility
11. Pathogenesis - molecular mechanisms, pathways
12. Macroscopic appearance - gross morphology
13. Histopathology - microscopic features, cellular composition
14. Immunophenotype - immunohistochemical markers
15. Differential diagnosis - diseases to differentiate
16. Cytology - cytological features
17. Diagnostic molecular pathology - molecular testing
18. Essential and desirable diagnostic criteria - diagnostic requirements
19. Grading / Staging - WHO grade, staging criteria
20. Prognosis and prediction - survival, recurrence factors

{already_extracted_section}

## STRICT RULES
1. Extract ONLY content that EXCLUSIVELY belongs to "{section_name}"
2. DO NOT include content that belongs to other sections listed above
3. DO NOT repeat any content that has already been extracted (see above)
4. If a sentence mentions MRI/CT findings, it belongs to "Imaging", NOT other sections
5. Keep citation markers like [1], [2] unchanged
6. If no NEW content exclusively belongs to this section, return empty string

## Medical Report Content
{content}

## Return Format
Return ONLY the extracted content (with citations). Return empty string if nothing found.
No explanations, no formatting, no section headers.
DO NOT include any content that was already extracted for previous sections."""

EXTRACT_REFERENCES_PROMPT = """Extract the reference list from the end of this medical report.

## Medical Report Content
{content}

## Task
Find the References/Bibliography section at the end and extract all numbered references.

## Return Format (strict JSON)
{{
    "1": "URL or citation text for reference 1",
    "2": "URL or citation text for reference 2"
}}

Return {{}} if no references found. Return JSON only, no explanation."""

EXTRACT_ID_PROMPT = """Extract the main disease/tumor name from this report and convert to ID format.

## Report Content (first 1000 chars)
{content}

## Task
1. Find the main disease/tumor name (usually in the title or first paragraph)
2. Convert to lowercase ID format: spaces → underscores, remove special characters

## Return
Return ONLY the ID string, e.g.: dysplastic_cerebellar_gangliocytoma

No explanation."""


# ============================================================================
# Table Data Structure
# ============================================================================

@dataclass
class ReportTable:
    """Report table with progressive filling."""
    id: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize all Sections as empty
        if not self.sections:
            self.sections = {name: "" for name, _ in SECTIONS}
    
    def set_section(self, section_name: str, content: str):
        """Fill content for a specific Section."""
        if section_name in self.sections:
            self.sections[section_name] = content.strip()
    
    def get_filled_count(self) -> int:
        """Get the number of filled Sections."""
        return sum(1 for v in self.sections.values() if v.strip())
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "sections": self.sections,
            "references": self.references,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string (single line)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def print_status(self):
        """Print current table filling status."""
        filled = self.get_filled_count()
        total = len(SECTIONS)
        print(f"\n📊 Table filling status: {filled}/{total}")
        for name, _ in SECTIONS:
            status = "✅" if self.sections.get(name, "").strip() else "⬜"
            content_preview = self.sections.get(name, "")[:50]
            if content_preview:
                content_preview = content_preview.replace("\n", " ") + "..."
            print(f"  {status} {name}: {content_preview}")


# ============================================================================
# Utility Functions
# ============================================================================

def parse_json_response(response: str) -> dict:
    """Parse JSON returned by LLM."""
    if not response:
        return {}
    
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE
    )
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================================
# Progressive Table Filler Class
# ============================================================================

class ReportTableFiller:
    """Progressive table filler converter."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # Track already extracted content to avoid duplication
        self._extracted_sections: dict[str, str] = {}
        
        # Token usage statistics
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0
    
    def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call LLM."""
        messages = [
            {"role": "system", "content": "You are a precise medical content extractor. Follow instructions exactly. Return only what is asked, no extra text."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            
            # Collect token usage
            self.total_calls += 1
            if hasattr(resp, 'usage') and resp.usage:
                self.prompt_tokens += resp.usage.prompt_tokens or 0
                self.completion_tokens += resp.usage.completion_tokens or 0
            
            return resp.choices[0].message.content or ""
        except Exception as e:
            if self.verbose:
                print(f"  ❌ LLM call failed: {e}")
            return ""
    
    def get_token_stats(self) -> dict:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_calls": self.total_calls,
        }
    
    def reset_token_stats(self) -> None:
        """Reset token statistics."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0

    def _extract_id(self, content: str) -> str:
        """Extract report ID."""
        if self.verbose:
            print("📝 Extracting report ID...")
        
        prompt = EXTRACT_ID_PROMPT.format(content=content[:2000])
        response = self._call_llm(prompt, max_tokens=100)
        
        # Clean response
        report_id = response.strip().lower()
        report_id = re.sub(r'[^a-z0-9_]', '', report_id.replace(' ', '_'))
        
        if not report_id:
            report_id = "unknown_report"
        
        if self.verbose:
            print(f"  ✅ ID: {report_id}")
        
        return report_id
    
    def _build_already_extracted_prompt(self) -> str:
        """Build prompt section showing already extracted content."""
        if not self._extracted_sections:
            return ""
        
        lines = ["## ALREADY EXTRACTED CONTENT (DO NOT REPEAT ANY OF THIS)"]
        for section_name, content in self._extracted_sections.items():
            if content.strip():
                # Truncate long content for prompt efficiency
                truncated = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"### {section_name}:")
                lines.append(truncated)
                lines.append("")
        
        return "\n".join(lines)
    
    def _extract_section(self, content: str, section_name: str, section_desc: str) -> str:
        """Extract content for a single Section with strict boundaries."""
        already_extracted = self._build_already_extracted_prompt()
        
        prompt = SECTION_EXTRACT_PROMPT.format(
            section_name=section_name,
            section_desc=section_desc,
            content=content,
            already_extracted_section=already_extracted,
        )
        
        response = self._call_llm(prompt)
        extracted = response.strip()
        
        # Post-processing: remove any content that exactly matches previous sections
        if extracted:
            for prev_section, prev_content in self._extracted_sections.items():
                if prev_content and prev_content.strip():
                    # Remove exact matches
                    if prev_content.strip() in extracted:
                        extracted = extracted.replace(prev_content.strip(), "").strip()
                    # Also check for sentence-level duplicates
                    prev_sentences = set(s.strip() for s in prev_content.split('.') if len(s.strip()) > 20)
                    for prev_sent in prev_sentences:
                        if prev_sent in extracted:
                            extracted = extracted.replace(prev_sent, "").strip()
        
        # Clean up any resulting empty lines or double spaces
        extracted = re.sub(r'\n\s*\n', '\n', extracted)
        extracted = re.sub(r'  +', ' ', extracted)
        extracted = extracted.strip()
        
        return extracted
    
    def _extract_references(self, content: str) -> dict[str, str]:
        """Extract reference list."""
        if self.verbose:
            print("📚 Extracting reference list...")
        
        prompt = EXTRACT_REFERENCES_PROMPT.format(content=content)
        response = self._call_llm(prompt)
        refs = parse_json_response(response)
        
        if self.verbose:
            print(f"  ✅ Extracted {len(refs)} references")
        
        return refs
    
    def fill_table(self, content: str, table: ReportTable | None = None) -> ReportTable:
        """
        Progressively fill the table.
        
        Args:
            content: Medical report text
            table: Optional existing table (supports resuming)
        
        Returns:
            Filled table
        """
        # Initialize table
        if table is None:
            table = ReportTable()
        
        # Reset extracted sections tracking
        self._extracted_sections = {}
        
        # 1. Extract ID (if not already done)
        if not table.id:
            table.id = self._extract_id(content)
        
        # 2. Fill each Section progressively
        total_sections = len(SECTIONS)
        for i, (section_name, section_desc) in enumerate(SECTIONS):
            # Skip already filled Sections
            if table.sections.get(section_name, "").strip():
                if self.verbose:
                    print(f"[{i+1}/{total_sections}] ⏭️  {section_name} (already filled, skipping)")
                # Still track it for deduplication
                self._extracted_sections[section_name] = table.sections[section_name]
                continue
            
            if self.verbose:
                print(f"[{i+1}/{total_sections}] 📝 Extracting {section_name}...")
            
            section_content = self._extract_section(content, section_name, section_desc)
            
            # Track this content for future deduplication
            if section_content:
                self._extracted_sections[section_name] = section_content
            
            table.set_section(section_name, section_content)
            
            if self.verbose:
                if section_content:
                    preview = section_content[:80].replace("\n", " ")
                    print(f"  ✅ Filled ({len(section_content)} chars): {preview}...")
                else:
                    print(f"  ⬜ No relevant content found")
        
        # 3. Extract references (if not already done)
        if not table.references:
            table.references = self._extract_references(content)
        
        return table


# ============================================================================
# File Processing
# ============================================================================

def fill_table_from_file(
    input_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> ReportTable:
    """Read report from file and fill table."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if verbose:
        print(f"📄 Reading report: {input_path}")
    
    with input_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    if verbose:
        print(f"   Report length: {len(content)} characters")
    
    # Create filler
    filler = ReportTableFiller(
        model=model,
        base_url=base_url,
        api_key=api_key,
        verbose=verbose,
    )
    
    # Fill table
    table = filler.fill_table(content)
    
    # Print status
    if verbose:
        table.print_status()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(table.to_json() + "\n")
    
    if verbose:
        print(f"\n💾 Saved to: {output_path}")
    
    return table


def fill_tables_from_directory(
    input_dir: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    pattern: str = "*.md",
    verbose: bool = True,
) -> list[ReportTable]:
    """Process all reports in a directory."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Clear output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")
    
    files = sorted(input_dir.glob(pattern))
    if verbose:
        print(f"📁 Found {len(files)} files\n")
    
    results = []
    for i, fp in enumerate(files):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(files)}] Processing: {fp.name}")
            print('='*60)
        
        try:
            table = fill_table_from_file(
                str(fp), str(output_path),
                model=model, base_url=base_url, api_key=api_key,
                verbose=verbose,
            )
            results.append(table)
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ Completed {len(results)}/{len(files)} files")
        print('='*60)
    
    return results