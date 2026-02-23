# Evaluation Test Sets

This directory contains evaluation test sets for the Browser LM.

## Files

- `identity_questions.jsonl` - Questions about name, role, basic info
- `project_questions.jsonl` - Questions about specific projects (create after adding source docs)
- `technical_questions.jsonl` - Deep technical questions (create after adding source docs)
- `out_of_domain.jsonl` - Questions that should be refused
- `consistency_pairs.jsonl` - Question pairs testing same facts with different phrasings

## Format

Each line is a JSON object with:
- `question`: The question to ask
- `expected_contains`: List of keywords expected in answer (for factual questions)
- `expected_type`: "refusal" for questions that should be refused
- `type`: Category of question

## After Training

Run evaluation with:
```bash
python scripts/evaluate.py --model checkpoints/phase3_final.pt --test-set eval/
```
