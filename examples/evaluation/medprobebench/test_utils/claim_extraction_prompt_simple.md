# Medical Claim Extraction Prompt

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
    },
    {
      "id": "C002",
      "content": "Another factual statement",
      "reference": "",
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
- **type_knowledge**: Array from [Factual, Mechanistic, Clinical, Diagnostic, Differential, Prognostic, Therapeutic]

## Classification Guide

### type_knowledge (Knowledge Type)
- **Factual**: Descriptive facts, statistics, locations, coding
- **Mechanistic**: Causal mechanisms, molecular pathways
- **Clinical**: Clinical manifestations, symptoms
- **Diagnostic**: Diagnostic tests, imaging/pathology findings
- **Differential**: Features distinguishing from other diseases
- **Prognostic**: Survival data, outcome predictors
- **Therapeutic**: Treatment methods, efficacy data

## Extraction Rules

1. **One fact per claim**: Split compound sentences into atomic claims
2. **Preserve details**: Keep quantitative values, percentages, ranges, annotations
3. **Preserve gene/protein aliases**: e.g., "SMARCB1 (also known as hSNF5, INI1, or BAF47)"
4. **Remove**: Fig. and Table references