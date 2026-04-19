"""
Generate medical guideline reports with OpenAI GPT.
Supports Web Search if the selected model allows it.
"""

import json
import time
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI, APIError


@dataclass
class OpenAIConfig:
    """OpenAI configuration."""
    model: str = "gpt-4o"
    max_tokens: int = 16000
    temperature: float = 0
    enable_search: bool = True  # Enable Web Search (requires model support)


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def update(self, usage):
        if usage:
            self.prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self.completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
            self.total_tokens += getattr(usage, 'total_tokens', 0) or 0
    
    def to_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class OpenAISearchGenerator:
    """Generate medical guideline reports using OpenAI."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        config: OpenAIConfig = None,
        verbose: bool = True,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        
        # Build client arguments
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        self.config = config or OpenAIConfig()
        self.verbose = verbose
        self.max_retries = max_retries
        self.token_usage = TokenUsage()
    
    def _save_partial_result(self, prompt_file: Path, output_path: Path, content: str, error: str):
        """Save partially generated content."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"<!-- INCOMPLETE: Generation failed -->\n")
            f.write(f"<!-- Error: {error} -->\n\n")
            f.write(content)
        
        partial_result = {
            "sample_id": prompt_file.stem,
            "content": content,
            "status": "incomplete",
            "error": str(error),
            "timestamp": time.time(),
        }
        with open(output_path.with_suffix('.partial.json'), 'w', encoding='utf-8') as f:
            json.dump(partial_result, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"   ⚠️  Saved partial result ({len(content)} chars)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIError, ConnectionError, TimeoutError)),
        reraise=True,
        before_sleep=lambda retry_state: print(f"   🔄 Retry {retry_state.attempt_number}/3")
    )
    def _create_completion(self, request_params: dict):
        """Internal API call with retry support."""
        return self.client.chat.completions.create(**request_params)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIError, ConnectionError, TimeoutError)),
        reraise=True
    )
    def generate_from_prompt_file(self, prompt_file: Path, output_path: Path = None) -> dict:
        """Generate a report from a prompt file."""
        start_time = time.perf_counter()
        prompt_content = prompt_file.read_text(encoding="utf-8")
        sample_id = prompt_file.stem
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 Generating: {prompt_file.name}")
            print(f"   Model: {self.config.model}")
            print(f"   Web Search: {'Enabled' if self.config.enable_search else 'Disabled'}")
            if self.base_url:
                print(f"   Base URL: {self.base_url}")
                print(f"   Base Key: {self.api_key[:10]}****")
            print(f"{'='*60}\n")
        
        partial_content = ""
        try:
            # Build request parameters
            request_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt_content}],
            }
            
            # Add web_search tool when search is enabled (Responses API format)
            # Note: this applies to models that support web search (e.g., gpt-4o-search-preview)
            if self.config.enable_search:
                # For models supporting web_search, use the tools parameter
                # Different API versions may require different formats
                if "search" in self.config.model.lower():
                    # Use the web_search_preview tool in the Responses API
                    request_params["tools"] = [{"type": "web_search_preview"}]
            
            # Call with retry wrapper
            response = self._create_completion(request_params)
            
            # Extract content as early as possible
            if response.choices and response.choices[0].message.content:
                partial_content = response.choices[0].message.content
            
            # Update token usage stats
            if response.usage:
                self.token_usage.update(response.usage)
            
            # Extract content
            content = ""
            tool_calls = []
            annotations = []
            
            if response.choices:
                choice = response.choices[0]
                message = choice.message
                
                # Extract text content
                if message.content:
                    content = message.content
                
                # Extract tool call information
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name if tool_call.function else None,
                                "arguments": tool_call.function.arguments if tool_call.function else None,
                            }
                        })
                
                # Check for annotations (some models return search citations this way)
                if hasattr(message, 'annotations') and message.annotations:
                    for ann in message.annotations:
                        annotations.append({
                            "type": getattr(ann, 'type', 'unknown'),
                            "text": getattr(ann, 'text', ''),
                            "url": getattr(ann, 'url', '') if hasattr(ann, 'url') else '',
                        })
            
            elapsed_time = time.perf_counter() - start_time
            
            # Detect whether search was used
            search_tools = [t for t in tool_calls if "search" in t.get("function", {}).get("name", "").lower()]
            has_search = len(search_tools) > 0 or len(annotations) > 0
            
            # Extract citations from content (OpenAI search results often use [number] format)
            import re
            citation_pattern = r'\[(\d+)\]'
            citations = re.findall(citation_pattern, content)
            unique_citations = list(set(citations))
            
            if self.verbose:
                if has_search or unique_citations:
                    print(f"   🔍 Search/Citations detected!")
                    if tool_calls:
                        print(f"      - Tool calls: {len(tool_calls)}")
                    if annotations:
                        print(f"      - Annotations: {len(annotations)}")
                    if unique_citations:
                        print(f"      - Citation refs: {len(unique_citations)}")
                else:
                    print(f"   ⚠️  No search/citations detected")
            
            # Get token usage
            token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if response.usage:
                token_info = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            
            result = {
                "sample_id": sample_id,
                "prompt_file": str(prompt_file),
                "content": content,
                "model": self.config.model,
                "elapsed_time": elapsed_time,
                "search_info": {
                    "enabled": self.config.enable_search,
                    "has_search": has_search or len(unique_citations) > 0,
                    "tool_calls": tool_calls,
                    "annotations": annotations,
                    "citation_count": len(unique_citations),
                },
                "token_usage": token_info,
                "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                "api_config": {
                    "base_url": self.base_url,
                    "model": self.config.model,
                }
            }
            
            if self.verbose:
                print(f"\n✅ Done! Time: {elapsed_time:.1f}s, Chars: {len(content):,}")
            
            if output_path:
                self._save_result(result, output_path)
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error after all retries: {e}")
                print(f"   Partial content length: {len(partial_content)}")
            
            if output_path and partial_content:
                self._save_partial_result(prompt_file, output_path, partial_content, str(e))
            
            raise
    
    def _save_result(self, result: dict, output_path: Path):
        """Save generation result."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(result.get("content", ""))
        
        # Save search metadata
        if result.get("search_info", {}).get("annotations"):
            search_path = output_path.parent / f"{output_path.stem}_search.json"
            with open(search_path, 'w', encoding='utf-8') as f:
                json.dump(result["search_info"], f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"   💾 Saved: {output_path}")
    
    def get_token_stats(self) -> dict:
        return self.token_usage.to_dict()
    
    def reset_token_stats(self):
        self.token_usage = TokenUsage()


def get_prompt_files(prompt_dir: Path, version: str = "v1") -> list[Path]:
    """Get prompt file list for a specific version."""
    version_dir = prompt_dir / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {version_dir}")
    return sorted(version_dir.glob("*.md"))


def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Generate medical guideline reports using OpenAI")
    
    script_dir = Path(__file__).parent
    parser.add_argument("--prompt-dir", type=str, default=str(script_dir / "generated_prompts"))
    parser.add_argument("--prompt-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("-o", "--output", type=str, default=str(script_dir / "openai_output"))
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-search", action="store_true", help="Disable Web Search")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please provide API key via --api-key or OPENAI_API_KEY")
        return
    
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    
    config = OpenAIConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_search=not args.no_search,
    )
    generator = OpenAISearchGenerator(
        api_key=api_key,
        base_url=base_url,
        config=config,
        verbose=not args.quiet
    )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {output_dir}")
    
    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.exists():
        print(f"❌ Prompt directory not found: {prompt_dir}")
        return
    
    try:
        prompt_files = get_prompt_files(prompt_dir, args.prompt_version)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    if args.sample_id:
        prompt_files = [f for f in prompt_files if f.stem == args.sample_id]
    if args.max_samples:
        prompt_files = prompt_files[:args.max_samples]
    
    if not prompt_files:
        print("❌ No prompt files found")
        return
    
    print(f"📝 Processing {len(prompt_files)} files with {args.model} (max_workers={args.max_workers})")
    print(f"🔍 Web Search: {'Enabled' if config.enable_search else 'Disabled'}")
    
    results = []
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(
                generator.generate_from_prompt_file,
                prompt_file,
                output_dir / f"{prompt_file.stem}.json"
            ): prompt_file
            for prompt_file in prompt_files
        }
        
        for i, future in enumerate(as_completed(future_to_file), 1):
            prompt_file = future_to_file[future]
            print(f"\n[{i}/{len(prompt_files)}] {prompt_file.stem}")
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"   ❌ Failed after all retries: {e}")
                failed_files.append(str(prompt_file))
    
    # Save summary
    if results:
        summary_path = output_dir / "summary.json"
        total_with_search = sum(r.get("search_info", {}).get("has_search", False) for r in results)
        total_citations = sum(r.get("search_info", {}).get("citation_count", 0) for r in results)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "num_samples": len(results),
                "total_tokens": generator.get_token_stats(),
                "api_config": {
                    "base_url": base_url,
                    "model": args.model,
                },
                "search_statistics": {
                    "search_enabled": config.enable_search,
                    "samples_with_search": total_with_search,
                    "total_citations": total_citations,
                    "search_usage_rate": f"{total_with_search/len(results)*100:.1f}%"
                },
                "results": [{
                    "sample_id": r["sample_id"],
                    "elapsed_time": r["elapsed_time"],
                    "has_search": r.get("search_info", {}).get("has_search", False),
                    "citation_count": r.get("search_info", {}).get("citation_count", 0),
                } for r in results],
                "failed_files": failed_files
            }, f, indent=2)
        print(f"\n✅ Summary: {summary_path}")
        print(f"🔍 Search usage: {total_with_search}/{len(results)} samples")
        print(f"📚 Total citations: {total_citations}")
    
    if failed_files:
        print(f"\n❌ Failed files ({len(failed_files)}): {failed_files}")
    
    stats = generator.get_token_stats()
    print(f"\n📊 Tokens: {stats['total_tokens']:,} (prompt: {stats['prompt_tokens']:,}, completion: {stats['completion_tokens']:,})")
    print(f"✅ Processed: {len(results)}/{len(prompt_files)}")


if __name__ == "__main__":
    main()
