# ConflictBench Fictional v1

A fully synthetic, platform-independent benchmark for conflicting-evidence retrieval and attribution.

## Easy
Intentionally hard:
- 15 questions
- 12 documents/question
- 1 active authoritative gold document
- 8 plausible conflicting documents
- 3 same-topic noise documents

## Hard
Intentionally super hard:
- 15 questions
- 20 documents/question
- 1 active authoritative gold document
- 14 plausible conflicting documents
- 5 same-topic noise documents
- strong majority pressure toward an incorrect answer

All entities, policies, units, and answers are fictional so pretrained world knowledge cannot supply the answer.

Conflict documents include stale references, drafts, wrong-scope material, and secondary summaries.
The gold document explicitly identifies itself as active and authoritative and as superseding older guidance.

Upload only one pack at a time. Do not upload questions.csv, ground_truth_manifest.csv, or results_template.csv.
Keep the model, system prompt, tools, skills, and retrieval settings fixed across Easy and Hard.
