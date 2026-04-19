"""
GuideBench Global Scorer

Evaluates generated medical guideline content quality independently,
without comparing to reference content (to save tokens).

Dimensions (from global_eval):
- Comprehensiveness (28%)
- Insight Depth (28%)
- Accuracy Standards (26%)
- Readability Utility (18%)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from abc import ABC, abstractmethod

from openai import AsyncOpenAI, OpenAI


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""
    criterion: str
    explanation: str
    weight: float
    score: Optional[float]  # Now can be None if evaluation failed
    reasoning: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Check if this score is valid (not None)."""
        return self.score is not None


@dataclass
class DimensionScore:
    """Score for an evaluation dimension."""
    dimension: str
    weight: float
    criterion_scores: List[CriterionScore]
    
    @property
    def weighted_score(self) -> Optional[float]:
        """Calculate weighted score for this dimension (normalized to 0-1)."""
        valid_scores = [cs for cs in self.criterion_scores if cs.is_valid]
        if not valid_scores:
            return None  # Return None if no valid scores
        
        total = sum(cs.score * cs.weight for cs in valid_scores)
        total_weight = sum(cs.weight for cs in valid_scores)
        return (total / total_weight) / 10.0 if total_weight > 0 else None
    
    @property
    def raw_score(self) -> Optional[float]:
        """Calculate raw average score for this dimension (0-10)."""
        valid_scores = [cs.score for cs in self.criterion_scores if cs.is_valid]
        if not valid_scores:
            return None
        return sum(valid_scores) / len(valid_scores)
    
    @property
    def num_valid_scores(self) -> int:
        """Number of valid scores in this dimension."""
        return sum(1 for cs in self.criterion_scores if cs.is_valid)


@dataclass
class GlobalScore:
    """Complete global evaluation score."""
    sample_id: str
    dimension_scores: Dict[str, DimensionScore]
    dimension_weights: Dict[str, float]
    elapsed_time: float = 0.0
    grader_calls: int = 0
    
    @property
    def final_score(self) -> Optional[float]:
        """Calculate final weighted score across all dimensions (0-1 scale)."""
        total = 0.0
        total_weight = 0.0
        
        for dim_name, dim_score in self.dimension_scores.items():
            weight = self.dimension_weights.get(dim_name, 0.0)
            weighted = dim_score.weighted_score
            
            if weighted is not None:
                total += weighted * weight
                total_weight += weight
        
        # Return None if no valid dimensions, otherwise normalize by actual weight
        return total / total_weight if total_weight > 0 else None
    
    @property
    def is_valid(self) -> bool:
        """Check if this evaluation has at least some valid scores."""
        return self.final_score is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "final_score": self.final_score,
            "dimension_scores": {
                dim_name: {
                    "dimension": dim_score.dimension,
                    "weight": dim_score.weight,
                    "weighted_score": dim_score.weighted_score,
                    "raw_score": dim_score.raw_score,
                    "criterion_scores": [
                        {
                            "criterion": cs.criterion,
                            "explanation": cs.explanation,
                            "weight": cs.weight,
                            "score": cs.score,
                            "reasoning": cs.reasoning
                        }
                        for cs in dim_score.criterion_scores
                    ]
                }
                for dim_name, dim_score in self.dimension_scores.items()
            },
            "dimension_weights": self.dimension_weights,
            "elapsed_time": self.elapsed_time,
            "grader_calls": self.grader_calls,
        }


# ============================================================================
# Optimized LLM Prompt - Evaluate quality without reference comparison
# ============================================================================

GLOBAL_EVAL_PROMPT = """You are a senior medical guideline reviewer with extensive experience in evidence-based medicine and clinical guideline evaluation.

You routinely assess guidelines for publication, regulatory review, or clinical adoption. Your default evaluation stance is skeptical, conservative, and risk-aware.

## Scoring Philosophy (MUST FOLLOW):
- 9–10: Exceptional, publication-ready quality with strong evidence, clear methodology, and no meaningful weaknesses. These scores should be extremely rare.
- 7–8: Good quality but with clear and non-trivial limitations.
- 5–6: Average or marginal quality with notable deficiencies.
- 3–4: Serious methodological or clinical problems.
- 0–2: Fundamentally flawed, unsafe, or poorly justified.

## Strict Rules:
- Be strict and conservative at all times.
- Missing, vague, or weakly justified information MUST reduce the score.
- Do NOT reward confident tone, fluent language, or medical-sounding terminology.
- If a criterion is only partially satisfied, the score MUST be 6 or lower.
- When in doubt, choose the LOWER score.
- Assume this evaluation will be used to filter out low-quality or unsafe guidelines.

## Content to Evaluate:
{content}

## Criteria to Score:
{criteria_list}

## Output Format (STRICT):
You MUST respond with ONLY a valid JSON object.
No explanations, no markdown, no extra text.

The JSON must have this exact structure:
{{"scores": [{{"criterion": "<criterion name>", "score": <integer 0-10>, "reason": "<brief, critical justification>"}}]}}

## Example Output (for format reference only — do NOT imitate scores):
{{"scores": [
  {{"criterion": "Section Coverage", "score": 6, "reason": "Covers main topics but omits dosage rationale and contraindications"}},
  {{"criterion": "Topic Depth", "score": 5, "reason": "Provides high-level discussion without detailed clinical justification"}}
]}}

Now evaluate the content and return JSON only:"""

# ============================================================================
# Optimized LLM Prompt - RELATIVE evaluation against GT reference
# ============================================================================

GLOBAL_EVAL_PROMPT_RELATIVE = """You are a senior medical guideline reviewer conducting a STRICT BENCHMARK evaluation.

Your task is to score a GENERATED medical guideline by DIRECTLY COMPARING it to a REFERENCE guideline (ground truth).

## Reference Guideline (Ground Truth, Score = 10):
{reference_content}

## Generated Guideline (To Be Scored by Deviation):
{generated_content}

## Evaluation Criteria (Each scored independently, GT = 10):
{criteria_list}

IMPORTANT ANCHOR RULE:
- The REFERENCE guideline represents perfect, expert-level quality and is implicitly scored as **10/10**.
- The GENERATED guideline is scored ONLY by how much it DEVIATES from the reference.
- Scores MUST be assigned by starting from 10 and subtracting points based on differences, weaknesses, or omissions.

## Relative Scoring Philosophy (GT-ANCHORED):
- **10**: Generated guideline is effectively identical to the reference (or equivalent in all clinically meaningful content)
- **8–9**: Minor deviations from the reference (small omissions, reduced detail, or wording differences without clinical impact)
- **6–7**: Noticeable deviations (missing rationale, reduced specificity, or weaker guidance in several areas)
- **4–5**: Major deviations (important sections missing, oversimplified guidance, or reduced clinical safety)
- **2–3**: Severe deviations (critical content missing, inaccuracies, or potentially unsafe recommendations)
- **0–1**: Extreme deviation (fundamentally incorrect, misleading, or largely unrelated to the reference)

## Mandatory Benchmark Rules (NON-NEGOTIABLE):
1. **GT = 10 by definition** — do NOT independently judge quality
2. **Score ONLY the distance between GENERATED and REFERENCE**
3. **Any missing, weaker, or less specific content MUST reduce the score**
4. **Vagueness counts as deviation** when the reference is specific or evidence-based
5. **Equal section titles do NOT imply equivalence** — content depth and precision matter
6. **When uncertain, subtract points** — never inflate scores in benchmark settings

## Output Requirements (STRICT):
- Output ONLY a valid JSON object
- No explanations, no markdown, no additional text
- Use EXACTLY the following structure:

{{"scores": [
  {{"criterion": "<criterion name>", "score": <integer 0-10>, "reason": "<explicit difference from GT explaining point deductions>"}}
]}}

## Example (FORMAT ONLY):
{{"scores": [
  {{"criterion": "Clinical Specificity", "score": 7, "reason": "GT specifies exact thresholds and dosages; generated omits numerical criteria"}},
  {{"criterion": "Safety Considerations", "score": 5, "reason": "GT details contraindications and monitoring; generated mentions safety only in general terms"}}
]}}

Now score the GENERATED guideline by measuring its deviation from the REFERENCE and return JSON only:"""


class BaseCriterionEvaluator(ABC):
    """Abstract base class for criterion evaluation."""
    
    @abstractmethod
    async def evaluate_batch(
        self,
        criterions: List[Dict[str, Any]],
        generated_content: str,
    ) -> List[Tuple[float, str]]:
        """Evaluate multiple criteria in one call to save tokens."""
        pass

class LLMEvaluator(BaseCriterionEvaluator):
    """LLM-based evaluator - evaluates all criteria in one call to save tokens."""
    
    def __init__(
        self, 
        client: AsyncOpenAI,
        model_name: str = "gpt-4o-mini",
        max_concurrent: int = 10,
        verbose: bool = False,
        max_retries: int = 2,
        use_json_mode: bool = True,  # New parameter to enable JSON mode
    ):
        self.client = client
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.max_retries = max_retries
        self.use_json_mode = use_json_mode
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Token usage
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    async def _balanced_truncate(self, content: str, max_length: int) -> str:
        """
        Truncate content using balanced sampling (front + middle + end).
        
        This preserves the overall document structure better than simple truncation.
        """
        if not content or len(content) <= max_length:
            return content
        
        # Split into 3 parts: 40% front, 30% middle, 30% end
        front_ratio, middle_ratio, end_ratio = 0.45, 0.45, 0.1
        
        front_len = int(max_length * front_ratio)
        middle_len = int(max_length * middle_ratio)
        end_len = max_length - front_len - middle_len  # Ensure exact total
        
        content_len = len(content)
        
        # Extract parts
        front_part = content[:front_len]
        
        # Middle part: center of the document
        middle_start = (content_len - middle_len) // 2
        middle_part = content[middle_start:middle_start + middle_len]
        
        # End part
        end_part = content[-end_len:]
        
        # Combine with markers for clarity
        truncated = (
            front_part + 
            "\n\n[... content truncated ...]\n\n" + 
            middle_part + 
            "\n\n[... content truncated ...]\n\n" + 
            end_part
        )
        
        return truncated
    
    async def evaluate_batch(
        self,
        criterions: List[Dict[str, Any]],
        generated_content: str,
        reference_content: str = "",  # New parameter
    ) -> List[Tuple[Optional[float], str]]:
        """Evaluate all criteria in ONE LLM call to save tokens."""
        
        # Build criteria list for prompt - use numbered format for clarity
        criteria_text = "\n".join([
            f"{i+1}. {crit.get('criterion', 'Unknown')}: {crit.get('explanation', '')[:150]}"
            for i, crit in enumerate(criterions)
        ])
        
        # Truncate content to save tokens
        # generated_truncated = generated_content[:30000] if len(generated_content) > 10000 else generated_content
        # reference_truncated = reference_content[:30000] if len(reference_content) > 10000 else reference_content
        
        generated_truncated = await self._balanced_truncate(generated_content, 30000)
        reference_truncated = await self._balanced_truncate(reference_content, 30000)
        
        # print("generated_truncated is:", generated_truncated)
        # print("reference_truncated is:", reference_truncated)
        
        # Choose prompt template based on whether reference is provided
        if reference_content.strip():
            prompt = GLOBAL_EVAL_PROMPT_RELATIVE.format(
                reference_content=reference_truncated,
                generated_content=generated_truncated,
                criteria_list=criteria_text,
            )
        else:
            # Fallback to absolute evaluation if no reference
            prompt = GLOBAL_EVAL_PROMPT.format(
                content=generated_truncated,
                criteria_list=criteria_text,
            )
        
        # Use a local variable to control JSON mode for the current call, avoiding modification of instance state
        current_use_json_mode = self.use_json_mode
        
        async with self._semaphore:
            # Try multiple times with retry
            for attempt in range(self.max_retries + 1):
                try:
                    # Build request parameters
                    request_params = {
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are a JSON-only medical evaluator. Always respond with valid JSON matching the requested format. Never include markdown code blocks or explanations."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,  # Changed from 0.1 to 0 for deterministic results
                        "max_tokens": 800,  # Increased for more criteria
                    }
                    
                    # Add JSON mode if supported and enabled
                    if current_use_json_mode:
                        # OpenAI's JSON mode - ensures valid JSON output
                        request_params["response_format"] = {"type": "json_object"}
                    
                    response = await self.client.chat.completions.create(**request_params)
                    
                    result = response.choices[0].message.content or ""
                    
                    # Collect token usage
                    if hasattr(response, 'usage') and response.usage:
                        self.prompt_tokens += response.usage.prompt_tokens or 0
                        self.completion_tokens += response.usage.completion_tokens or 0
                    
                    # Try to parse
                    parsed_results = self._parse_batch_response(result, criterions)
                    
                    # Check if parsing was successful
                    failed_count = sum(1 for score, reason in parsed_results 
                                      if score is None or "Parse failed" in reason or "No match" in reason)
                    
                    # If more than half succeeded, accept the results
                    if failed_count < len(criterions) / 2:
                        return parsed_results
                    
                    # If this was not the last attempt and parsing failed, retry
                    if attempt < self.max_retries:
                        if self.verbose:
                            print(f"  ⚠️ Parse failed ({failed_count}/{len(criterions)}) on attempt {attempt + 1}, retrying...")
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    
                    # Last attempt - return what we have
                    if self.verbose:
                        print(f"  ⚠️ All {self.max_retries + 1} attempts completed, {failed_count} criteria failed")
                    
                    return parsed_results
                    
                except Exception as e:
                    error_msg = str(e)
                    if self.verbose:
                        print(f"  ⚠️ LLM evaluation failed (attempt {attempt + 1}): {error_msg[:100]}")
                    
                    # Check if it's a JSON mode not supported error
                    # Only modify the local variable, not the instance state
                    if "json" in error_msg.lower() and current_use_json_mode:
                        if self.verbose:
                            print(f"  ℹ️ JSON mode not supported, retrying without it...")
                        current_use_json_mode = False
                        continue
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    
                    # Final attempt failed with exception
                    return [(None, f"Evaluation failed: {error_msg[:50]}") for _ in criterions]
            
            # Should not reach here
            return [(None, "Evaluation failed: unknown error") for _ in criterions]
    
    def _parse_batch_response(
        self, 
        response: str, 
        criterions: List[Dict[str, Any]]
    ) -> List[Tuple[Optional[float], str]]:
        """Parse batch response from LLM with robust error handling."""
        
        # Strategy 1: Try standard JSON parsing
        try:
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip(), flags=re.IGNORECASE)
            match = re.search(r'\{[\s\S]*\}', cleaned)
            if match:
                data = json.loads(match.group(0))
                return self._extract_scores_from_data(data, criterions)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  ⚠️ JSON decode failed (Strategy 1): {e}")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Strategy 1 failed: {e}")
        
        # Strategy 2: Try to fix common JSON issues
        try:
            fixed_response = self._fix_common_json_issues(response)
            data = json.loads(fixed_response)
            return self._extract_scores_from_data(data, criterions)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Strategy 2 (JSON fix) failed: {e}")
        
        # Strategy 3: Try regex extraction as last resort
        try:
            return self._extract_scores_with_regex(response, criterions)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Strategy 3 (regex) failed: {e}")
        
        # All strategies failed - log raw response for debugging
        if self.verbose:
            print(f"  ⚠️ All parsing strategies failed. Raw response (first 500 chars):")
            print(f"     {response[:500]}")
        
        # Return default scores with more informative message
        return [(None, f"Parse failed: all strategies exhausted") for _ in criterions]
    
    def _extract_scores_from_data(
        self, 
        data: Dict[str, Any], 
        criterions: List[Dict[str, Any]]
    ) -> List[Tuple[Optional[float], str]]:
        """Extract scores from parsed JSON data."""
        scores_list = data.get("scores", [])
        
        if not scores_list:
            raise ValueError("No 'scores' field in response")
        
        # Build mapping from criterion names to scores (case-insensitive, fuzzy)
        score_map = {}
        for s in scores_list:
            criterion_name = s.get("criterion", "").lower().strip()
            if criterion_name:
                score_map[criterion_name] = s
        
        results = []
        for crit in criterions:
            crit_name = crit.get("criterion", "").lower().strip()
            
            # Try exact match first
            if crit_name in score_map:
                matched = score_map[crit_name]
            else:
                # Try fuzzy match (substring search both ways)
                matched = None
                for key, value in score_map.items():
                    if crit_name in key or key in crit_name:
                        matched = value
                        break
            
            if matched:
                try:
                    score = float(matched.get("score", 5))
                    score = max(0.0, min(10.0, score))
                    reason = str(matched.get("reason", ""))[:200]
                    results.append((score, reason))
                except (ValueError, TypeError) as e:
                    if self.verbose:
                        print(f"  ⚠️ Invalid score format for '{crit_name}': {e}")
                    results.append((None, f"Invalid score format: {e}"))
            else:
                if self.verbose:
                    print(f"  ⚠️ No match found for criterion '{crit_name}'")
                results.append((None, f"No match in response for '{crit_name}'"))
        
        return results
    
    def _fix_common_json_issues(self, response: str) -> str:
        """Attempt to fix common JSON formatting issues from LLM responses."""
        # Remove markdown code blocks
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip(), flags=re.IGNORECASE | re.MULTILINE)
        
        # Extract JSON object
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            raise ValueError("No JSON object found in response")
        
        json_str = match.group(0)
        
        # Fix common issues:
        # 1. Trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 2. Single quotes instead of double quotes (careful with apostrophes in text)
        # Only replace single quotes around keys and simple values
        json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)
        
        # 3. Unescaped quotes in strings (basic fix)
        # This is tricky - skip for now to avoid breaking valid JSON
        
        return json_str
    
    def _extract_scores_with_regex(
        self, 
        response: str, 
        criterions: List[Dict[str, Any]]
    ) -> List[Tuple[Optional[float], str]]:
        """Extract scores using regex as fallback strategy."""
        results = []
        
        # Pattern: "criterion": "name", "score": N, "reason": "text"
        pattern = r'"criterion"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"reason"\s*:\s*"([^"]*)"'
        matches = re.finditer(pattern, response, re.IGNORECASE)
        
        score_map = {}
        for match in matches:
            criterion_name = match.group(1).lower().strip()
            score = float(match.group(2))
            reason = match.group(3)
            score_map[criterion_name] = (score, reason)
        
        if not score_map:
            raise ValueError("No scores extracted via regex")
        
        # Match to criterions
        for crit in criterions:
            crit_name = crit.get("criterion", "").lower().strip()
            
            if crit_name in score_map:
                score, reason = score_map[crit_name]
                score = max(0.0, min(10.0, score))
                results.append((score, reason[:200]))
            else:
                # Fuzzy match
                matched = None
                for key, value in score_map.items():
                    if crit_name in key or key in crit_name:
                        matched = value
                        break
                
                if matched:
                    score, reason = matched
                    score = max(0.0, min(10.0, score))
                    results.append((score, reason[:200]))
                else:
                    results.append((None, f"Regex: no match for '{crit_name}'"))
        
        return results


class GlobalEvaluator:
    """
    GuideBench Global Evaluator - Optimized version with RELATIVE scoring.
    
    Evaluates generated content quality RELATIVE TO reference content.
    Uses batch evaluation to minimize LLM calls and token usage.
    """
    
    def __init__(
        self,
        client: AsyncOpenAI | OpenAI,
        grader_model: str = "gpt-4o-mini",
        use_llm: bool = True,
        max_concurrent: int = 10,
        verbose: bool = False,
        use_json_mode: bool = True,  # New parameter
    ):
        self.grader_model = grader_model
        self.use_llm = use_llm
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.use_json_mode = use_json_mode
        
        # Ensure AsyncOpenAI
        if isinstance(client, OpenAI):
            self.client = AsyncOpenAI(
                api_key=client.api_key,
                base_url=str(client.base_url) if client.base_url else None,
            )
        else:
            self.client = client
        
        # Initialize evaluator
        if use_llm:
            self.evaluator = LLMEvaluator(
                client=self.client,
                model_name=grader_model,
                max_concurrent=max_concurrent,
                verbose=verbose,
                use_json_mode=use_json_mode,
            )
    
    async def evaluate(
        self,
        sample: Dict[str, Any],
        generated_content: str,
        reference_content: str = "",  # New parameter
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample's global score RELATIVE to reference.
        
        Args:
            sample: GT sample data with global_eval field
            generated_content: Generated guideline content
            reference_content: Reference guideline content (GT)
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.perf_counter()
        sample_id = sample.get("id", "unknown")
        global_eval = sample.get("global_eval", {})
        
        dimension_weights = global_eval.get("dimension_weight", {})
        criterions = global_eval.get("criterions", {})
        
        if not criterions:
            # Use default criteria if not specified
            criterions = self._get_default_criteria()
            dimension_weights = {
                "Comprehensiveness": 0.28,
                "Insight_Depth": 0.28,
                "Accuracy_Standards": 0.26,
                "Readability_Utility": 0.18,
            }
        
        # Extract reference content from sample if not provided
        if not reference_content.strip():
            reference_content = sample.get("content", "") or sample.get("report", "")
        
        if self.verbose:
            print(f"\n📊 Global Evaluation [{sample_id}] (RELATIVE mode)")
            print(f"   Dimensions: {list(dimension_weights.keys())}")
            print(f"   Reference length: {len(reference_content)} chars")
            print(f"   Generated length: {len(generated_content)} chars")
        
        dimension_scores = {}
        total_grader_calls = 0
        
        # Evaluate each dimension with batch call
        for dim_name, dim_criterions in criterions.items():
            if self.verbose:
                print(f"   Processing dimension: {dim_name} with {len(dim_criterions)} criteria")
            if not dim_criterions:
                continue
            
            # Batch evaluate all criteria in this dimension (1 LLM call per dimension)
            results = await self.evaluator.evaluate_batch(
                dim_criterions, 
                generated_content,
                reference_content,  # Pass reference content
            )
            total_grader_calls += 1  # One call per dimension
            
            # Build criterion scores - handle None scores
            crit_scores = []
            for crit, (score, reason) in zip(dim_criterions, results):
                crit_scores.append(CriterionScore(
                    criterion=crit.get("criterion", ""),
                    explanation=crit.get("explanation", ""),
                    weight=crit.get("weight", 1.0 / len(dim_criterions)),
                    score=score,  # Can be None now
                    reasoning=reason,
                ))
            
            dimension_scores[dim_name] = DimensionScore(
                dimension=dim_name,
                weight=dimension_weights.get(dim_name, 0.0),
                criterion_scores=crit_scores,
            )
        
        global_score = GlobalScore(
            sample_id=sample_id,
            dimension_scores=dimension_scores,
            dimension_weights=dimension_weights,
            elapsed_time=time.perf_counter() - start_time,
            grader_calls=total_grader_calls,
        )
        
        # Collect token stats
        token_usage = {}
        if self.use_llm and isinstance(self.evaluator, LLMEvaluator):
            token_usage = {
                "prompt_tokens": self.evaluator.prompt_tokens,
                "completion_tokens": self.evaluator.completion_tokens,
                "total_tokens": self.evaluator.prompt_tokens + self.evaluator.completion_tokens,
            }
        
        if self.verbose:
            final = global_score.final_score
            if final is not None:
                print(f"   Final score: {final:.3f}")
            else:
                print(f"   Final score: INVALID (no valid dimension scores)")
            
            for dim_name, dim_score in dimension_scores.items():
                weighted = dim_score.weighted_score
                raw = dim_score.raw_score
                num_valid = dim_score.num_valid_scores
                if weighted is not None:
                    print(f"     - {dim_name}: {weighted:.3f} (raw: {raw:.1f}/10, {num_valid}/{len(dim_score.criterion_scores)} valid)")
                else:
                    print(f"     - {dim_name}: INVALID (0/{len(dim_score.criterion_scores)} valid)")
        
        # Return 0.0 instead of None for compatibility, but add validity flag
        final_score_value = global_score.final_score if global_score.final_score is not None else 0.0
        
        return {
            "global_score": final_score_value,  # Always a float (0.0 if invalid)
            "is_valid": global_score.is_valid,  # Flag to indicate validity
            "dimension_scores": {
                dim_name: {
                    "weighted_score": dim_score.weighted_score if dim_score.weighted_score is not None else 0.0,
                    "raw_score": dim_score.raw_score if dim_score.raw_score is not None else 0.0,
                    "weight": dim_score.weight,
                    "num_valid": dim_score.num_valid_scores,
                    "num_total": len(dim_score.criterion_scores),
                }
                for dim_name, dim_score in dimension_scores.items()
            },
            "detailed_scores": global_score.to_dict(),
            "elapsed_time": global_score.elapsed_time,
            "grader_calls": total_grader_calls,
            "stats": {
                "grader_calls": total_grader_calls,
                "grader_total_time": global_score.elapsed_time,
                "token_usage": token_usage,
            }
        }
    
    async def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        generated_contents: Dict[str, str],
        reference_contents: Dict[str, str] = None,  # New parameter
    ) -> Dict[str, Any]:
        """Batch evaluate multiple samples."""
        results = []
        
        for sample in samples:
            sample_id = sample.get("id", "")
            if sample_id in generated_contents:
                # Get reference content (from dict or sample)
                ref_content = ""
                if reference_contents and sample_id in reference_contents:
                    ref_content = reference_contents[sample_id]
                else:
                    ref_content = sample.get("content", "") or sample.get("report", "")
                
                result = await self.evaluate(
                    sample, 
                    generated_contents[sample_id],
                    ref_content,  # Pass reference content
                )
                result["sample_id"] = sample_id
                results.append(result)
        
        if not results:
            return {"error": "No samples evaluated"}
        
        # Aggregate statistics - filter out invalid results
        valid_results = [r for r in results if r.get("is_valid", False)]
        invalid_count = len(results) - len(valid_results)
        
        if valid_results:
            final_scores = [r["global_score"] for r in valid_results]
            
            dimension_aggregates = {}
            if valid_results:
                dim_names = valid_results[0].get("dimension_scores", {}).keys()
                for dim_name in dim_names:
                    # Only aggregate from valid results
                    dim_scores = [
                        r["dimension_scores"][dim_name]["weighted_score"] 
                        for r in valid_results 
                        if dim_name in r.get("dimension_scores", {}) 
                        and r["dimension_scores"][dim_name].get("num_valid", 0) > 0
                    ]
                    if dim_scores:
                        dimension_aggregates[dim_name] = {
                            "mean": sum(dim_scores) / len(dim_scores),
                            "min": min(dim_scores),
                            "max": max(dim_scores),
                        }
            
            return {
                "num_samples": len(results),
                "num_valid": len(valid_results),
                "num_invalid": invalid_count,
                "global_score": {
                    "mean": sum(final_scores) / len(final_scores),
                    "min": min(final_scores),
                    "max": max(final_scores),
                },
                "dimension_scores": dimension_aggregates,
                "individual_results": results,
            }
        else:
            return {
                "num_samples": len(results),
                "num_valid": 0,
                "num_invalid": len(results),
                "error": "All evaluations failed",
                "global_score": {
                    "mean": 0.0,  # Return 0.0 instead of None
                    "min": 0.0,
                    "max": 0.0,
                },
                "individual_results": results,
            }

    def _get_default_criteria(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return default evaluation criteria if not specified in sample."""
        print("   ⚠️ No criteria specified in sample; using default criteria.")
        return {
            "Comprehensiveness": [
                {"criterion": "Section Coverage", "explanation": "Covers all key medical guideline sections", "weight": 0.5},
                {"criterion": "Topic Depth", "explanation": "Sufficient detail on each topic", "weight": 0.5},
            ],
            "Insight_Depth": [
                {"criterion": "Medical Accuracy", "explanation": "Uses correct medical terminology and facts", "weight": 0.5},
                {"criterion": "Evidence Quality", "explanation": "References credible sources", "weight": 0.5},
            ],
            "Accuracy_Standards": [
                {"criterion": "Terminology Precision", "explanation": "Uses precise medical coding (ICD, WHO)", "weight": 0.5},
                {"criterion": "Factual Correctness", "explanation": "Statements are factually accurate", "weight": 0.5},
            ],
            "Readability_Utility": [
                {"criterion": "Writing Clarity", "explanation": "Clear and concise writing", "weight": 0.5},
                {"criterion": "Organization", "explanation": "Well-organized structure", "weight": 0.5},
            ],
        }


# ============================================================================
# CLI
# ============================================================================

import argparse

def main():
    parser = argparse.ArgumentParser(description="GuideBench Global Scorer (Optimized)")
    
    repo_root = Path(__file__).resolve().parents[2]
    default_dataset = repo_root / "datasets" / "guidebench" / "test.jsonl"
    
    parser.add_argument("--dataset", "-d", type=str, default=str(default_dataset))
    parser.add_argument("--generated", "-g", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="guidebench_global_scores.json")
    parser.add_argument("--grader-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use-llm", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.no_llm:
        args.use_llm = False
    
    asyncio.run(main_async(args))


async def main_async(args):
    """Async main function."""
    import os
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url if base_url else None)
    
    evaluator = GlobalEvaluator(
        client=client,
        grader_model=args.grader_model,
        use_llm=args.use_llm,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose,
    )
    
    # Load dataset
    samples = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    # Load generated content
    generated_contents = {}
    with open(args.generated, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sample_id = item.get("id", "")
                content = item.get("content", "") or item.get("report", "") or item.get("output", "")
                if sample_id and content:
                    generated_contents[sample_id] = content
    
    print(f"Loaded {len(generated_contents)} generated contents")
    
    if args.max_examples:
        samples = samples[:args.max_examples]
    
    # Evaluate
    results = await evaluator.evaluate_batch(samples, generated_contents)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output}")
    
    print(f"\nAggregate Scores:")
    if results.get("global_score"):
        print(f"  Global Score: {results['global_score']['mean']:.3f}")
        for dim_name, dim_stats in results.get('dimension_scores', {}).items():
            print(f"  {dim_name}: {dim_stats['mean']:.3f}")

if __name__ == "__main__":
    main()
