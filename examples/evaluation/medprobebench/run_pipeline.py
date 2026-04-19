"""
Integration process script:
1. check_format: Check the format of all MD files
2. report_to_table: Perform structured conversion on files that fail the check
3. standardize_md: Standardize the section names of all MD files
4. table_to_md: Output the conversion results as a standard format MD file
5. extract_claims: Extract claims from the standard MD file
6. enrich_references: Add URLs to the references in the claims
7. convert_to_jsonl_format: Convert the extracted claims to JSONL format
Final output: Consistent format MD file + claims JSONL file
"""

import argparse
import shutil
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

SCRIPT_DIR = Path(__file__).parent.resolve()
TEST_UTILS_DIR = SCRIPT_DIR / "test_utils"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TEST_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_UTILS_DIR))

from test_utils.check_format import check_format, iter_md_files, MATCH_THRESHOLD
from test_utils.report_to_table import ReportTableFiller
from test_utils.table_to_md import entry_to_md
from test_utils.enrich_references import ReferenceEnricher, EnrichmentStats
from test_utils.extract_claims import (
    process_guideline,
    load_prompt_template,
    get_token_stats as ec_get_token_stats,
    reset_token_stats as ec_reset_token_stats,
)
import test_utils.extract_claims as ec_module
from test_utils.convert_to_jsonl_format import convert_json_to_jsonl
from test_utils.standardize_md import standardize_file


def load_reference_config() -> dict:
    """Load the reference_config.json configuration file."""
    config_path = TEST_UTILS_DIR / "reference_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@dataclass
class PipelineTokenStats:
    """Pipeline token usage statistics."""
    report_to_table_tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0, "calls": 0})
    extract_claims_tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0, "calls": 0})
    total_tokens: int = 0
    
    def add_report_to_table(self, prompt: int, completion: int, calls: int = 1):
        self.report_to_table_tokens["prompt"] += prompt
        self.report_to_table_tokens["completion"] += completion
        self.report_to_table_tokens["calls"] += calls
        self.total_tokens += prompt + completion
    
    def add_extract_claims(self, prompt: int, completion: int, calls: int = 1):
        self.extract_claims_tokens["prompt"] += prompt
        self.extract_claims_tokens["completion"] += completion
        self.extract_claims_tokens["calls"] += calls
        self.total_tokens += prompt + completion
    
    def to_dict(self) -> dict:
        return {
            "report_to_table": {
                "prompt_tokens": self.report_to_table_tokens["prompt"],
                "completion_tokens": self.report_to_table_tokens["completion"],
                "total_tokens": self.report_to_table_tokens["prompt"] + self.report_to_table_tokens["completion"],
                "calls": self.report_to_table_tokens["calls"],
            },
            "extract_claims": {
                "prompt_tokens": self.extract_claims_tokens["prompt"],
                "completion_tokens": self.extract_claims_tokens["completion"],
                "total_tokens": self.extract_claims_tokens["prompt"] + self.extract_claims_tokens["completion"],
                "calls": self.extract_claims_tokens["calls"],
            },
            "total": {
                "prompt_tokens": self.report_to_table_tokens["prompt"] + self.extract_claims_tokens["prompt"],
                "completion_tokens": self.report_to_table_tokens["completion"] + self.extract_claims_tokens["completion"],
                "total_tokens": self.total_tokens,
                "total_calls": self.report_to_table_tokens["calls"] + self.extract_claims_tokens["calls"],
            },
        }
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("📊 Token Usage Summary")
        print("=" * 60)
        
        r2t = self.report_to_table_tokens
        if r2t["calls"] > 0:
            print(f"  report_to_table:")
            print(f"    - Calls: {r2t['calls']}")
            print(f"    - Prompt tokens: {r2t['prompt']:,}")
            print(f"    - Completion tokens: {r2t['completion']:,}")
            print(f"    - Total: {r2t['prompt'] + r2t['completion']:,}")
        
        ec = self.extract_claims_tokens
        if ec["calls"] > 0:
            print(f"  extract_claims:")
            print(f"    - Calls: {ec['calls']}")
            print(f"    - Prompt tokens: {ec['prompt']:,}")
            print(f"    - Completion tokens: {ec['completion']:,}")
            print(f"    - Total: {ec['prompt'] + ec['completion']:,}")
        
        total_calls = r2t["calls"] + ec["calls"]
        total_prompt = r2t["prompt"] + ec["prompt"]
        total_completion = r2t["completion"] + ec["completion"]
        
        print(f"\n  TOTAL:")
        print(f"    - Calls: {total_calls}")
        print(f"    - Prompt tokens: {total_prompt:,}")
        print(f"    - Completion tokens: {total_completion:,}")
        print(f"    - Total tokens: {self.total_tokens:,}")


# Global stats tracker
_pipeline_stats = PipelineTokenStats()


def check_all_files(input_path: Path, threshold: float, workers: int = 4) -> Tuple[List[Path], List[Path]]:
    """Check all files and return lists of passed and failed files (parallel version)."""
    md_files = list(iter_md_files(input_path))
    passed = []
    failed = []
    
    print("=" * 60)
    print("Step 1: Format Check")
    print("=" * 60)
    print(f"  Concurrent workers: {workers}")
    
    lock = Lock()
    
    def check_single_file(md_file):
        is_valid, details = check_format(md_file, threshold)
        status = "✓ Passed" if is_valid else "✗ Failed"
        with lock:
            print(f"  {md_file.name}: {status} (match ratio: {details['match_ratio']:.0%})")
        return md_file, is_valid
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_single_file, f) for f in md_files]
        
        for future in as_completed(futures):
            md_file, is_valid = future.result()
            if is_valid:
                passed.append(md_file)
            else:
                failed.append(md_file)
    
    print(f"\n  Summary: {len(passed)} passed, {len(failed)} failed")
    return passed, failed


def convert_failed_files(
    failed_files: List[Path],
    temp_jsonl: Path,
    model: str,
    base_url: str | None,
    api_key: str | None,
    verbose: bool = True,
    workers: int = 10,
) -> List[dict]:
    """Convert failed files to structured tables (parallel version)."""
    global _pipeline_stats
    
    if not failed_files:
        return []
    
    print("\n" + "=" * 60)
    print("Step 2: Structured Conversion (report_to_table)")
    print("=" * 60)
    print(f"  Concurrent workers: {workers}")
    
    temp_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp_jsonl.write_text("")
    
    tables = []
    lock = Lock()
    file_lock = Lock()
    
    def process_single_file(args):
        i, md_file = args
        filler = ReportTableFiller(
            model=model,
            base_url=base_url,
            api_key=api_key,
            verbose=False,  # Suppress verbose output for individual files
        )
        
        try:
            with lock:
                print(f"\n[{i+1}/{len(failed_files)}] Processing: {md_file.name}")
            
            content = md_file.read_text(encoding="utf-8")
            table = filler.fill_table(content)
            
            original_id = md_file.stem.lower().replace(" ", "_").replace("-", "_")
            
            if not table.id or table.id == "unknown_report":
                if verbose:
                    with lock:
                        print(f"  ⚠️  Model did not generate a valid ID, using original filename: {original_id}")
                table.id = original_id
            elif table.id != original_id:
                if verbose:
                    with lock:
                        print(f"  ⚠️  Model-generated ID ({table.id}) does not match original filename")
                        print(f"      Forcing original filename: {original_id}")
                table.id = original_id
            
            with file_lock:
                with temp_jsonl.open("a", encoding="utf-8") as f:
                    f.write(table.to_json() + "\n")
            
            with lock:
                print(f"  ✅ Done: {table.id} ({table.get_filled_count()}/20 sections)")
            
            return table.to_dict(), filler.get_token_stats()
            
        except Exception as e:
            with lock:
                print(f"  ❌ Failed: {e}")
            return None, None
    
    total_prompt = 0
    total_completion = 0
    total_calls = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_file, (i, f)) 
                  for i, f in enumerate(failed_files)]
        
        for future in as_completed(futures):
            result, token_stats = future.result()
            if result:
                tables.append(result)
            if token_stats:
                total_prompt += token_stats["prompt_tokens"]
                total_completion += token_stats["completion_tokens"]
                total_calls += token_stats["total_calls"]
    
    _pipeline_stats.add_report_to_table(total_prompt, total_completion, total_calls)
    
    return tables


def generate_md_from_converted(
    converted_tables: List[dict],
    output_dir: Path,
    verbose: bool = True,
) -> List[Path]:
    """Step 3: Generate MD files from conversion results (table_to_md)."""
    print("\n" + "=" * 60)
    print("Step 3: Generate MD Files (table_to_md)")
    print("=" * 60)
    
    if not converted_tables:
        print("  ⚠️  No conversion results to generate")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for entry in converted_tables:
        entry_to_md(entry, output_dir)
        tumor_id = entry.get("id", "unknown")
        generated_files.append(output_dir / f"{tumor_id}.md")
        if verbose:
            print(f"  ✓ Generated: {tumor_id}.md")
    
    print(f"\n  Total generated: {len(generated_files)} MD files -> {output_dir.resolve()}")
    return generated_files


def standardize_all_files(
    passed_files: List[Path],
    generated_files: List[Path],
    output_dir: Path,
    verbose: bool = True,
    workers: int = 10,
) -> List[Path]:
    """Step 4: Standardize section names of all MD files (passed + generated) to first-level headings (parallel version)."""
    print("\n" + "=" * 60)
    print("Step 4: Standardize Section Names (standardize_md)")
    print("=" * 60)
    
    all_files = passed_files + generated_files
    
    if not all_files:
        print("  ⚠️  No files to standardize")
        return []
    
    print(f"  Standardizing {len(all_files)} MD files")
    print(f"    - Files that passed check: {len(passed_files)}")
    print(f"    - Converted/generated files: {len(generated_files)}")
    print(f"  Concurrent workers: {workers}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    standardized_files = []
    modified_count = 0
    lock = Lock()
    
    def standardize_single_file(md_file):
        output_file = output_dir / md_file.name
        stats = standardize_file(md_file, output_file, verbose=False)
        has_changes = len(stats.get("matched", [])) > 0
        
        if verbose:
            status = f"✓ Standardized: {md_file.name} ({stats.get('sections_with_content', 0)} sections)" if has_changes else f"✓ Processed: {md_file.name} (no content)"
            with lock:
                print(f"  {status}")
        
        return output_file, has_changes
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(standardize_single_file, f) for f in all_files]
        
        for future in as_completed(futures):
            output_file, has_changes = future.result()
            standardized_files.append(output_file)
            if has_changes:
                modified_count += 1
    
    print(f"\n  📊 Standardization stats:")
    print(f"     Total files: {len(all_files)}")
    print(f"     With content: {modified_count}")
    print(f"     Without content: {len(all_files) - modified_count}")
    
    return standardized_files


def extract_claims_from_md(
    model: str,
    md_dir: Path,
    output_claims_dir: Path,
    api_config: dict,
    verbose: bool = True,
    force_extract: bool = True,
    workers: int = 10,
) -> bool:
    """Step 5: Extract claims from MD files (parallel version)."""
    global _pipeline_stats
    
    print("\n" + "=" * 60)


    print("Step 5: Extract Claims (extract_claims)")
    print("=" * 60)
    
    ref_config = load_reference_config()
    gpt_config = ref_config.get("gpt", {})
    
    final_base_url = api_config.get("base_url") or gpt_config.get("base_url")
    final_api_key = api_config.get("api_key") or gpt_config.get("api_key")
    
    if not final_api_key:
        print("  ❌ API key not found, please check configuration")
        return False
    
    from openai import OpenAI
    ec_module.client = OpenAI(
        base_url=final_base_url,
        api_key=final_api_key
    )
    
    ec_reset_token_stats()
    
    if force_extract and output_claims_dir.exists():
        print(f"  🗑️  Clearing existing claims directory: {output_claims_dir}")
        shutil.rmtree(output_claims_dir)
    
    output_claims_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        prompt_template = load_prompt_template()
    except FileNotFoundError:
        print(f"  ⚠️  Prompt template file not found, using default template")
        prompt_template = None
    
    md_files = list(md_dir.glob("*.md"))
    print(f"  Found {len(md_files)} MD files")
    
    if not md_files:
        print("  ⚠️  No MD files found")
        return False
    
    processed_files = {f.stem for f in output_claims_dir.glob("*.json")}
    unprocessed_files = [f for f in md_files if f.stem not in processed_files]
    
    if not unprocessed_files:
        print("  ✅ All files already processed")
        return True
    
    print(f"  Processing {len(unprocessed_files)} files...")
    print(f"  Concurrent workers: {workers}")
    
    success_count = 0
    lock = Lock()
    
    def process_single_guideline(args):
        i, md_file = args
        if verbose:
            with lock:
                print(f"\n  [{i+1}/{len(unprocessed_files)}] {md_file.name}")
        
        try:
            result = process_guideline(model ,md_file, prompt_template, output_claims_dir)
            if result:
                if verbose:
                    with lock:
                        print(f"    ✅ Success")
                return True
        except Exception as e:
            with lock:
                print(f"    ❌ Failed: {e}")
        return False
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_guideline, (i, f)) 
                  for i, f in enumerate(unprocessed_files)]
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    try:
        ec_stats = ec_get_token_stats()
        _pipeline_stats.add_extract_claims(
            ec_stats.get("prompt_tokens", 0),
            ec_stats.get("completion_tokens", 0),
            ec_stats.get("total_calls", 0),
        )
    except Exception:
        pass
    
    print(f"\n  ✅ Claims extraction complete: {success_count}/{len(unprocessed_files)}")
    return success_count > 0


def enrich_claims_references(
    model: str,
    claims_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    workers: int = 10,
    resolve_publisher: bool = True,
    use_llm: bool = False,
    verbose: bool = True,
) -> EnrichmentStats:
    """Step 6: Add URLs to references in claims."""
    print("\n" + "=" * 60)
    print("Step 6: Reference URL Enrichment (enrich_references)")
    print("=" * 60)
    
    ref_config = load_reference_config()
    processing_config = ref_config.get("processing", {})
    actual_workers = processing_config.get("reference_level_workers", workers)
    
    if not claims_dir.exists():
        print(f"  ⚠️  Claims directory does not exist: {claims_dir}")
        return EnrichmentStats()
    
    json_files = list(claims_dir.glob("*.json"))
    if not json_files:
        print("  ⚠️  No claims files found")
        return EnrichmentStats()
    
    print(f"  Found {len(json_files)} claims files")
    print(f"  Concurrent workers: {actual_workers}")
    if use_llm:
        print("  🤖 Using LLM to extract URL information")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enricher = ReferenceEnricher(
        cache_dir=cache_dir,
        workers=actual_workers,
        resolve_publisher=resolve_publisher,
        use_llm=use_llm,
        llm_config={"model": model}
    )
    
    total_stats = EnrichmentStats()
    
    for json_file in json_files:
        output_file = output_dir / json_file.name
        stats = enricher.enrich_file(json_file, output_file)
        
        total_stats.total += stats.total
        total_stats.complete += stats.complete
        total_stats.partial += stats.partial
        total_stats.no_pmid += stats.no_pmid
        total_stats.failed += stats.failed
    
    enricher.cache.save()
    
    print(f"\n  📊 Reference enrichment stats:")
    print(f"     Total: {total_stats.total}")
    if total_stats.total > 0:
        print(f"     ✅ Complete: {total_stats.complete} ({total_stats.complete/total_stats.total*100:.1f}%)")
        print(f"     ⚠️  Partial: {total_stats.partial} ({total_stats.partial/total_stats.total*100:.1f}%)")
        print(f"     ⏭️  No PMID: {total_stats.no_pmid} ({total_stats.no_pmid/total_stats.total*100:.1f}%)")
        print(f"     ❌ Failed: {total_stats.failed}")
    
    return total_stats


def convert_claims_to_jsonl(
    claims_dir: Path,
    md_dir: Path,
    output_jsonl: Path,
    verbose: bool = True,
) -> bool:
    """Step 7: Convert claims to JSONL format."""
    print("\n" + "=" * 60)
    print("Step 7: Convert to JSONL Format (convert_to_jsonl_format)")
    print("=" * 60)
    
    json_files = list(claims_dir.glob("*.json"))
    print(f"  Found {len(json_files)} claims JSON files")
    
    if not json_files:
        print("  ⚠️  No claims files found, skipping conversion")
        return False
    
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    if output_jsonl.exists():
        output_jsonl.unlink()
    
    success_count = 0
    for json_file in json_files:
        md_file = md_dir / f"{json_file.stem}.md"
        
        try:
            if convert_json_to_jsonl(json_file, md_file, output_jsonl):
                success_count += 1
                if verbose:
                    print(f"  ✓ Converted: {json_file.stem}")
        except Exception as e:
            print(f"  ❌ Conversion failed {json_file.name}: {e}")
    
    print(f"\n  ✅ JSONL conversion complete: {success_count}/{len(json_files)}")
    print(f"  📁 Output file: {output_jsonl.resolve()}")
    return success_count > 0


def run_pipeline(
    input_path: str,
    output_dir: str,
    threshold: float = MATCH_THRESHOLD,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
    force_convert: bool = False,
    skip_claims: bool = False,
    skip_enrich: bool = False,
    output_jsonl: str | None = None,
    enrich_workers: int = 10,
    no_publisher: bool = False,
    use_llm: bool = False,
    force_extract: bool = True,
    workers: int = 10,
):
    """Run the complete pipeline."""
    global _pipeline_stats
    _pipeline_stats = PipelineTokenStats()
    
    ref_config = load_reference_config()
    gpt_config = ref_config.get("gpt", {})
    output_config = ref_config.get("output", {})
    
    final_base_url = base_url or gpt_config.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    final_api_key = api_key or gpt_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    temp_jsonl = output_dir / ".temp_converted.jsonl"
    temp_generated_dir = output_dir / ".temp_generated"
    
    standardized_md_dir = output_dir
    claims_dir = output_dir / output_config.get("input_dir", "extracted_claims")
    enriched_claims_dir = output_dir / output_config.get("output_dir", "enriched_claims")
    cache_dir = output_dir / output_config.get("cache_dir", "cache")
    
    if output_jsonl is None:
        output_jsonl = output_dir / "guidebench_pred.jsonl"
    else:
        output_jsonl = Path(output_jsonl)
    
    print(f"\n{'#' * 60}")
    print("# Guide Bench Complete Pipeline")
    print(f"# Input: {input_path}")
    print(f"# Output: {output_dir}")
    print(f"# Threshold: {threshold:.0%}")
    print(f"# Concurrent workers: {workers}")
    print(f"# Extract Claims: {'No' if skip_claims else 'Yes'}")
    print(f"# Enrich References: {'No' if skip_enrich else 'Yes'}")
    print(f"# Force re-extract: {'Yes' if force_extract else 'No'}")
    print(f"# API Base URL: {final_base_url or '(default)'}")
    print(f"{'#' * 60}\n")
    
    # Step 1: Format check (parallel)
    if force_convert:
        print("⚠️  Force conversion mode: skipping format check, converting all files")
        passed_files = []
        failed_files = list(iter_md_files(input_path))
    else:
        print(f"Checking file format... {input_path} ")
        passed_files, failed_files = check_all_files(input_path, threshold, workers=workers)
    
    # Step 2: Convert failed files (parallel)
    converted_tables = []
    if failed_files:
        if not final_api_key:
            print("\n⚠️  API key not provided, unable to perform conversion")
        else:
            converted_tables = convert_failed_files(
                failed_files, temp_jsonl, model, final_base_url, final_api_key, verbose, workers=workers
            )
    
    # Step 3: Generate MD files from conversion results
    generated_files = generate_md_from_converted(
        converted_tables=converted_tables,
        output_dir=temp_generated_dir,
        verbose=verbose,
    )
    
    # Step 4: Standardize all MD files (parallel)
    standardized_files = standardize_all_files(
        passed_files=passed_files,
        generated_files=generated_files,
        output_dir=standardized_md_dir,
        verbose=verbose,
        workers=workers,
    )
    
    # Clean up temporary files and directories
    if temp_jsonl.exists():
        temp_jsonl.unlink()
    if temp_generated_dir.exists():
        shutil.rmtree(temp_generated_dir)
    
    total_md_files = len(standardized_files)
    
    # Step 5 & 6 & 7: Claims extraction, reference enrichment, and conversion
    if not skip_claims:
        if not final_api_key:
            print("\n⚠️  API key not provided, skipping claims extraction")
        else:
            api_config = {
                "base_url": final_base_url,
                "api_key": final_api_key,
            }
            extract_success = extract_claims_from_md(
                model=model,
                md_dir=standardized_md_dir,
                output_claims_dir=claims_dir,
                api_config=api_config,
                verbose=verbose,
                force_extract=force_extract,
                workers=workers,  # Parallel extraction
            )
            
            existing_claims = list(claims_dir.glob("*.json")) if claims_dir.exists() else []
            
            if extract_success or existing_claims:
                if not skip_enrich:
                    enrich_claims_references(
                        model=model,
                        claims_dir=claims_dir,
                        output_dir=enriched_claims_dir,
                        cache_dir=cache_dir,
                        workers=enrich_workers,
                        resolve_publisher=not no_publisher,
                        use_llm=use_llm,
                        verbose=verbose,
                    )
                    
                    enriched_files = list(enriched_claims_dir.glob("*.json")) if enriched_claims_dir.exists() else []
                    final_claims_dir = enriched_claims_dir if enriched_files else claims_dir
                else:
                    final_claims_dir = claims_dir
                
                final_claims_files = list(final_claims_dir.glob("*.json")) if final_claims_dir.exists() else []
                if final_claims_files:
                    convert_claims_to_jsonl(
                        claims_dir=final_claims_dir,
                        md_dir=standardized_md_dir,
                        output_jsonl=output_jsonl,
                        verbose=verbose,
                    )
                else:
                    print("\n⚠️  No claims files available for JSONL conversion")
            else:
                print("\n⚠️  Claims extraction failed, skipping subsequent steps")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Complete!")
    print("=" * 60)
    print(f"  ✓ Passed check: {len(passed_files)} files")
    print(f"  ✓ Converted/generated: {len(generated_files)} files")
    print(f"  ✓ Standardized output: {total_md_files} files (unified to first-level headings)")
    print(f"  📁 Standardized MD output dir: {standardized_md_dir.resolve()}")
    if not skip_claims and final_api_key:
        print(f"  📁 Claims dir: {claims_dir.resolve()}")
        if not skip_enrich:
            print(f"  📁 Enriched claims dir: {enriched_claims_dir.resolve()}")
        print(f"  📁 JSONL file: {output_jsonl.resolve()}")
    
    _pipeline_stats.print_summary()
    
    stats_file = output_dir / "token_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(_pipeline_stats.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  📁 Token stats saved to: {stats_file}")
    
    return total_md_files

from dotenv import load_dotenv
def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Integration pipeline: Format Check -> Structured Conversion -> Standardize -> Generate MD -> Extract Claims -> Enrich References -> JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input MD file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--output-jsonl", default=None, help="Output JSONL file path")
    parser.add_argument("--use-llm-enrich", action="store_true", help="Use LLM for reference URL extraction")
    parser.add_argument("--force-extract", action="store_true", default=False,
                        help="Force re-extract claims (enabled by default)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel processing workers (default: 10)")
    parser.add_argument("-t", "--threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"Format match threshold (default: {MATCH_THRESHOLD})")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--force", action="store_true", help="Force convert all files")
    parser.add_argument("--skip-claims", action="store_true", help="Skip claims extraction step") #
    parser.add_argument("--skip-enrich", action="store_true", help="Skip reference enrichment step") #
    parser.add_argument("--enrich-workers", type=int, default=10, help="Reference enrichment concurrent workers")
    parser.add_argument("--no-publisher", action="store_true", help="Skip publisher URL resolution")
    args = parser.parse_args()
    
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    print("base_url:", base_url or "(default)")
    print("api_key:", f"{api_key[:15]} ******" if api_key else "(not set)")
    
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        verbose=not args.quiet,
        force_convert=args.force,
        skip_claims=args.skip_claims,
        skip_enrich=args.skip_enrich,
        output_jsonl=args.output_jsonl,
        enrich_workers=args.enrich_workers,
        no_publisher=args.no_publisher,
        use_llm=args.use_llm_enrich,
        force_extract=args.force_extract,
        workers=args.workers,  # Pass workers parameter
    )


if __name__ == "__main__":
    main()