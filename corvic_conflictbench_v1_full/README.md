# Corvic ConflictBench v1

Purpose
-------
A one-day diagnostic stress test for conflicting evidence in Corvic, derived from
Discern-and-Answer / MacNoise NQ evaluation examples.

Dataset
-------
- 15 questions
- 12 documents per question
- 180 uploadable Markdown documents total
- Per question:
  - 1 gold document
  - 8 conflicting documents
  - 3 same-topic noise documents

IMPORTANT
---------
Upload ONLY the files inside `corvic_upload/` to Corvic.
Do NOT upload `ground_truth_manifest.csv`, `questions.csv`, or `seed_metadata.json`,
because they reveal the correct source.

Recommended test
----------------
For each row in `questions.csv`:
1. Start a fresh Corvic conversation/thread.
2. Ask the question exactly as written.
3. Record Corvic's answer.
4. Record which source(s) Corvic cites.
5. Check whether the gold document was retrieved/cited.

Primary metrics
---------------
- Answer Accuracy
- Gold Citation Accuracy
- Gold Retrieval Recall (if retrieval trace is visible)

Failure decomposition
---------------------
- Gold not retrieved -> retrieval failure
- Gold retrieved but wrong answer/source selected -> evidence-selection/conflict-resolution failure
- Correct answer but wrong citation -> attribution failure

Construction
------------
The gold documents use clean contexts from the source dataset.
Conflict documents use MacNoise counterfactual contexts plus compact synthetic
conflict excerpts derived from those counterfactual passages.
Noise documents retain same-topic background while removing the literal gold
answer where possible.

Neutral filenames are intentional so the system cannot infer document role from
file naming.
