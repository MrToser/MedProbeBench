"""
Extract medical claims from guideline markdown files.

Supports two usage modes:
1. Standalone: python extract_claims.py (uses api_config.json)
2. Called by run_pipeline.py (client set externally)
"""

import json
import re
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import shared utilities
from md_utils import extract_references_section, parse_references

# Lazy import of OpenAI to avoid errors at module load time
OpenAI = None

# Global variables
client = None  # Can be set externally
_token_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "total_calls": 0
}

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """# Medical Claim Extraction Prompt

## Task
Extract Medical Claims (C) from medical guideline text. Each claim is a minimal, independently verifiable factual statement.

## Output Format
```json
{
  "claims": [
    {
      "id": "C001",
      "content": "Extracted factual statement",
      "reference": "[1]",
      "type_knowledge": "Factual",
      "section": "Definition"
    }
  ]
}
```

## Fields
- **id**: Sequential identifier (C001, C002, ...)
- **content**: The claim text (preserve original wording and annotations)
- **reference**: Citation numbers like "[1]", "[2, 3]", or "" if none
- **type_knowledge**: From [Factual, Mechanistic, Clinical, Diagnostic, Differential, Prognostic, Therapeutic]

## Extraction Rules
1. **One fact per claim**: Split compound sentences into atomic claims
2. **Preserve details**: Keep quantitative values, percentages, ranges, annotations
3. **Preserve gene/protein aliases**: e.g., "SMARCB1 (also known as hSNF5, INI1, or BAF47)"
4. **Remove**: Fig. and Table references

## Current Section Title: {section}

## Input Text:

{text}
"""


def _ensure_openai():
    """Ensure OpenAI is imported."""
    global OpenAI
    if OpenAI is None:
        from openai import OpenAI as _OpenAI
        OpenAI = _OpenAI


def get_client():
    """Get OpenAI client, supports lazy initialization."""
    global client
    if client is None:
        _ensure_openai()
        # Try to load from config file
        config_paths = [
            Path(__file__).parent / "api_config.json",
            Path(__file__).parent.parent / "api_config.json",
            Path("api_config.json"),
        ]
        
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        api_key = api_key or config.get("api_key")
                        base_url = base_url or config.get("base_url")
                    break
                except Exception:
                    pass
        
        if api_key:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("No API key found. Set OPENAI_API_KEY or provide api_config.json")
    
    return client


def get_token_stats() -> dict:
    """Get token usage statistics."""
    return _token_stats.copy()


def reset_token_stats():
    """Reset token statistics."""
    global _token_stats
    _token_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_calls": 0
    }


def _update_token_stats(response):
    """Update token statistics."""
    if hasattr(response, 'usage') and response.usage:
        _token_stats["prompt_tokens"] += response.usage.prompt_tokens or 0
        _token_stats["completion_tokens"] += response.usage.completion_tokens or 0
        _token_stats["total_tokens"] += response.usage.total_tokens or 0
        _token_stats["total_calls"] += 1


def load_prompt_template(prompt_file: str = None) -> str:
    """Load prompt template."""
    if prompt_file:
        path = Path(prompt_file)
        if path.exists():
            return path.read_text(encoding="utf-8")
    
    # Try default paths
    default_paths = [
        Path(__file__).parent / "claim_extraction_prompt_simple.md",
        Path(__file__).parent.parent / "scripts" / "claim_extraction_prompt_simple.md",
        Path("scripts/claim_extraction_prompt_simple.md"),
    ]
    
    for path in default_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    
    return DEFAULT_PROMPT_TEMPLATE


def preprocess_markdown(md_text: str) -> str:
    """Preprocess markdown: remove images and Fig references."""
    # Remove image references: ![](...)
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)

    # Remove Fig. captions and standalone Fig lines
    lines = md_text.split('\n')
    filtered_lines = [
        line for line in lines
        if not re.match(r'^\s*Fig\.', line, re.IGNORECASE)
    ]

    return '\n'.join(filtered_lines)


def split_by_sections(md_text: str) -> list:
    """Split markdown by # headers."""
    sections = []
    current_section = []
    current_title = "Introduction"

    lines = md_text.split('\n')
    for line in lines:
        if re.match(r'^#+\s+', line):
            if current_section:
                sections.append({
                    'title': current_title,
                    'content': '\n'.join(current_section).strip()
                })
            current_title = re.sub(r'^#+\s+', '', line).strip()
            current_section = []
        else:
            current_section.append(line)

    if current_section:
        sections.append({
            'title': current_title,
            'content': '\n'.join(current_section).strip()
        })

    return sections


def extract_references(md_text: str) -> dict:
    """
    Extract references - using shared utilities.

    Extraction Method:
    1. extract_references_section: Identifies and extracts the text block under the References header.
    2. parse_references: Parses the extracted text into a dictionary (e.g., {"1": "Citation info..."}).
    """
    _, ref_section = extract_references_section(md_text)
    # print("ref_section is:", repr(ref_section))
    if not ref_section:
        print("   ⚠️ No References section found")
        return {}
    
    references = parse_references(ref_section)
    # print("references parsed:", references)
    print(f"   📚 Extracted {len(references)} references")
    # assert 1==0
    if references:
        # Print first 3 references for debugging
        for i, (num, text) in enumerate(list(references.items())[:3]):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"      [{num}] {preview}")
    
    return references


def remove_references_section(md_text: str) -> str:
    """Remove references section - using shared utilities."""
    content_without_refs, _ = extract_references_section(md_text)
    return content_without_refs


def extract_claims_from_section(model: str, section_title: str, section_content: str, prompt_template: str, max_retries: int = 3) -> dict:
    """Call LLM to extract claims."""
    import time

    # Build prompt
    if "{section}" in prompt_template and "{text}" in prompt_template:
        full_prompt = prompt_template.format(section=section_title, text=section_content)
    else:
        full_prompt = f"""{prompt_template}

## Current Section Title: {section_title}

## Input Text:

{section_content}

## IMPORTANT
You MUST return a valid JSON object with this exact structure:
{{"claims": [...]}}

Return ONLY the JSON object, no markdown code blocks, no explanations."""

    for attempt in range(max_retries):
        try:
            # Try calling the API
            try:
                # First try with response_format
                response = get_client().chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0,
                    timeout=300,
                    response_format={"type": "json_object"}
                )
            except (TypeError, Exception) as e:
                # If response_format is not supported, retry without it
                if "response_format" in str(e) or "unexpected keyword" in str(e).lower():
                    print(f"   ⚠️ response_format not supported, retrying without it...")
                    response = get_client().chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0,
                        timeout=300,
                    )
                else:
                    raise
            
            _update_token_stats(response)
            
            # Get response content
            content = response.choices[0].message.content
            
            # Check if empty
            if not content or content.strip() == "":
                print(f"   ⚠️ Empty response for '{section_title}'")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                    continue
                return {"claims": []}
            
            # Debug: print first 200 chars of raw response
            print(f"   📝 Raw response preview for '{section_title}': {repr(content[:200])}")
            
            # Clean and parse JSON
            cleaned_content = content.strip()
            
            # Remove markdown code block markers
            import re
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_content)
            if code_block_match:
                cleaned_content = code_block_match.group(1).strip()
            
            # Try to find JSON object {...}
            json_match = re.search(r'\{[\s\S]*\}', cleaned_content)
            if json_match:
                cleaned_content = json_match.group(0)
            
            # Parse JSON
            try:
                result = json.loads(cleaned_content)
                
                # Ensure the 'claims' field exists
                if "claims" not in result:
                    print(f"   ⚠️ Response missing 'claims' field for '{section_title}'")
                    result = {"claims": []}
                
                print(f"   ✅ Parsed {len(result.get('claims', []))} claims for '{section_title}'")
                return result
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error for '{section_title}':")
                print(f"      Error: {e}")
                print(f"      Content length: {len(content)}")
                print(f"      Content: {repr(content[:500])}")
                
                # Check if the response appears truncated
                if len(content) < 50 or content.count('{') != content.count('}'):
                    print(f"      ⚠️ Response appears truncated!")
                
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                    continue
                return {"claims": []}
            
        except Exception as e:
            error_str = str(e)
            print(f"   ⚠️ Attempt {attempt + 1}/{max_retries} failed for '{section_title}': {error_str}")
            
            # Rate limit handling
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = (attempt + 1) * 30
                print(f"      Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep((attempt + 1) * 10)

    return {"claims": []}


def process_section(model: str , section: dict, prompt_template: str) -> dict:
    """Process a single section."""
    section_title = section['title']
    section_content = section['content']

    if not section_content.strip():
        return None

    # Limit content length to avoid token overflow
    max_content_length = 12000  # ~3000 tokens
    if len(section_content) > max_content_length:
        print(f"   ⚠️ Section '{section_title}' too long ({len(section_content)} chars), truncating...")
        section_content = section_content[:max_content_length] + "\n\n[Content truncated...]"

    result = extract_claims_from_section(model ,section_title, section_content, prompt_template)

    if result and 'claims' in result and len(result['claims']) > 0:
        for claim in result['claims']:
            claim['section'] = section_title
        return {
            'title': section_title,
            'claims': result['claims']
        }

    return None


def process_guideline(model: str, md_path: Path, prompt_template: str, output_dir: Path) -> Path:
    """
    Process a single guideline file (main interface called by run_pipeline.py).
    
    Args:
        md_path: Path to the MD file
        prompt_template: Prompt template string
        output_dir: Output directory
        
    Returns:
        Output file path, or None on failure
    """
    try:
        print(f"\n📄 Processing: {md_path.name}")

        md_text = md_path.read_text(encoding='utf-8')
        
        # Extract references
        references = extract_references(md_text)
        print(f"   Found {len(references)} references")

        # Remove references section and preprocess
        md_text = remove_references_section(md_text)
        md_text = preprocess_markdown(md_text)

        # Split into sections
        sections = split_by_sections(md_text)
        print(f"   Split into {len(sections)} sections")

        # Initialize output
        output_file = output_dir / f"{md_path.stem}.json"
        all_claims = []

        # Load existing progress if any
        processed_sections = set()
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    all_claims = existing_data.get('claims', [])
                    processed_sections = {c.get('section') for c in all_claims}
                    print(f"   Resuming from {len(all_claims)} existing claims")
            except Exception:
                pass

        # Filter sections to process
        sections_to_process = [
            s for s in sections
            if s['title'] not in processed_sections and s['content'].strip()
        ]

        if not sections_to_process:
            print(f"   ✅ All sections already processed")
            return output_file

        print(f"   Processing {len(sections_to_process)} sections...")

        # Process sections in parallel
        section_results = []
        max_section_workers = 3

        with ThreadPoolExecutor(max_workers=max_section_workers) as executor:
            future_to_section = {
                executor.submit(process_section, model, section, prompt_template): section
                for section in sections_to_process
            }

            with tqdm(total=len(sections_to_process), desc=f"   {md_path.stem}", leave=False) as pbar:
                for future in as_completed(future_to_section):
                    section = future_to_section[future]
                    try:
                        result = future.result()
                        if result:
                            section_results.append(result)
                            print(f"\n      ✅ Section '{result['title']}': {len(result['claims'])} claims")
                    except Exception as e:
                        print(f"\n      ❌ Section '{section['title']}' failed: {e}")
                    finally:
                        pbar.update(1)

        # Combine claims and assign IDs
        claim_counter = len(all_claims) + 1
        for section_result in section_results:
            for claim in section_result['claims']:
                claim['id'] = f"C{claim_counter:03d}"
                claim_counter += 1
                all_claims.append(claim)

        # Save results
        output_data = {
            'guideline': md_path.stem,
            'claims': all_claims,
            'reference': references
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n   ✅ Total claims extracted: {len(all_claims)}")
        return output_file

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None