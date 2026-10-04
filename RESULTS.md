# Protein Structure Detective results

Single-seed pilot; uncertainty is clustered by protein.

| Checkpoint | Split | Success | 95% interval | Episodes |
|---|---|---:|---|---:|
| evaluation-grpo-test | test | 98.0% | 95.0%–100.0% | 100 |
| evaluation-original-test | test | 0.0% | 0.0%–0.0% | 100 |
| evaluation-original-validation | validation | 0.0% | 0.0%–0.0% | 140 |
| evaluation-sft-test | test | 98.0% | 95.0%–100.0% | 100 |
| sft | validation | 100.0% | 100.0%–100.0% | 140 |

SFT improvement over original: 98.0 percentage points (paired protein-bootstrap interval 95.0 to 100.0).

Observed RL improvement: 0.0 percentage points.
Paired protein-bootstrap 95% interval: 0.0 to 0.0 points.
The 10-point target was not met.

## Sampled training-question probe (the objective RL optimizes)

Thirty fixed training questions, four samples each at temperature 0.8, identical seed before and after GRPO.

| Policy | Sampled success | Mixed groups | All-success groups | All-failure groups |
|---|---:|---:|---:|---:|
| SFT (before RL) | 94.2% | 13.3% | 86.7% | 0.0% |
| SFT + GRPO (after RL) | 99.2% | 3.3% | 96.7% | 0.0% |

Sampled success changed by 5.0 percentage points on these training questions. This is a training-set, sampled-decoding diagnostic, not a held-out result.

## Validation during GRPO (greedy, 140 held-out questions)

| Optimizer step | Success | Valid calls | Calls/episode |
|---:|---:|---:|---:|
| 25 | 100.0% | 1.0 | 2.80 |
| 50 | 100.0% | 1.0 | 2.80 |
| 75 | 100.0% | 1.0 | 2.80 |
| final | 100.0% | 1.0 | 2.80 |

The RL checkpoint was selected on validation success only; ties were broken toward the later checkpoint.

## Tool use and task breakdown

| Checkpoint | No-match accuracy | Valid calls | Calls/episode | Tokens/episode |
|---|---:|---:|---:|---:|
| evaluation-grpo-test | 100.0% | 0.9928057553956835 | 2.78 | 144.6 |
| evaluation-original-test | 0.0% | 0.3933333333333333 | 1.50 | 115.2 |
| evaluation-original-validation | 0.0% | 0.39565217391304347 | 1.64 | 111.7 |
| evaluation-sft-test | 100.0% | 0.9928057553956835 | 2.78 | 144.5 |
| sft | 100.0% | 1.0 | 2.80 | 143.3 |

## Compute spend

Conservative recorded estimate/reservations: $8.26. This is not confirmed provider billing.
Recorded jobs: 10. Rate assumptions and per-job durations are preserved in budget.json.

## Failure inspection

244 failed held-out episodes across all evaluated checkpoints; the first 20 are listed, trained checkpoints first.

- evaluation-grpo-test / P01709-04 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': 'BO'}`
- evaluation-grpo-test / P33919-01 (standard): wrong_protein; answer: `{'pdb_id': 'PDB:12345', 'resolution_angstrom': 2.33, 'status': 'no_match', 'uniprot_accession': 'PDB:12345'}`
- evaluation-sft-test / P01709-04 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': 'BO'}`
- evaluation-sft-test / P33919-01 (standard): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': 'PRD10000000'}`
- evaluation-original-test / O35324-00 (standard): nonqualifying_structure; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'found', 'uniprot_accession': 'O35324'}`
- evaluation-original-test / O35324-01 (standard): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / O35324-02 (standard): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / O35324-03 (standard): nonqualifying_structure; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'found', 'uniprot_accession': 'O35324'}`
- evaluation-original-test / O35324-04 (alias): generation_limit; answer: `None`
- evaluation-original-test / O35324-05 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / O35324-06 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / O35324-07 (accession): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / O35324-08 (no_match): generation_limit; answer: `None`
- evaluation-original-test / O35324-09 (no_match): nonqualifying_structure; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'found', 'uniprot_accession': 'O35324'}`
- evaluation-original-test / P01709-00 (standard): nonqualifying_structure; answer: `{'pdb_id': 'P01709', 'resolution_angstrom': 1.05, 'status': 'found', 'uniprot_accession': 'P01709'}`
- evaluation-original-test / P01709-01 (standard): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / P01709-02 (standard): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / P01709-03 (standard): nonqualifying_structure; answer: `{'pdb_id': 'P01709', 'resolution_angstrom': 1.95, 'status': 'found', 'uniprot_accession': 'P01709'}`
- evaluation-original-test / P01709-04 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`
- evaluation-original-test / P01709-05 (alias): wrong_protein; answer: `{'pdb_id': None, 'resolution_angstrom': None, 'status': 'no_match', 'uniprot_accession': None}`

# Original-model failure review

Reviewed 20 complete validation trajectories from W&B run `b015455fcb6f` before seeing test results. These are qualitative diagnoses of the observed messages, not additional reward categories. The original model failed all 140 validation questions: 95 wrong-protein answers, 36 nonqualifying structures, five malformed actions, and four generation-limit failures.

| Question | Observed failure |
|---|---|
| A8K830-00 | Guessed P04949 before identity lookup; later copied the correct UniProt accession into the PDB field. |
| A8K830-01 | Guessed P02109; included the organism in the protein search string; returned an unresolved null accession. |
| O00198-00 | Used Homo_sapiens as an accession; resolved the protein afterward but stopped with a found answer and null structure. |
| O00198-01 | Used the placeholder PDB:12345 as an accession, then searched with an overlong combined organism/name string. |
| P01860-00 | Invented IMMUNOGLOBIN_HCG3; after resolving P01860, used that accession as the PDB ID. |
| P01860-01 | Repeated the same unsuccessful placeholder-accession and combined-name searches; returned null accession. |
| P12838-00 | Used a protein-name slug as the accession; later found the accession but did not search structures again. |
| P12838-01 | Repeated placeholder PDB_12345 and unsuccessful combined-name queries. |
| P52129-00 | Used X as both an accession and the final PDB identifier, without evidence; copied the threshold into the resolution field. |
| P52129-01 | Passed organism plus protein name to the accession argument and interpreted an error as no match. |
| P52129-02 | Generated malformed JSON around the organism's parentheses after an invented accession failed. |
| P58400-00 | Invented NEU-1-B and returned no match with a null accession after the tool error. |
| P58400-03 | Resolved the protein after an invalid lookup, then copied its accession into the PDB field and the threshold into resolution. |
| P63235-00 | Used E. coli as an accession; later copied the resolved accession into the PDB field without inspection. |
| P63235-01 | Repeated searches for E. coli K12 as both accession and protein query; no resolved protein in the answer. |
| Q46865-00 | Used gene alias MqsR as an accession and then generated malformed organism JSON. |
| Q46865-01 | Resolved mqsr after using E. coli as an accession, but returned found with null structure fields. |
| Q46865-02 | Used MqsR directly as an accession and interpreted the lookup error as no match. |
| Q6UXT8-00 | Invented ALK-LTK-1 and failed to recover through protein search. |
| Q6UXT8-06 | Used placeholder X, then copied the resolved UniProt accession into the PDB field and the threshold into resolution. |

The dominant errors concern identifier types, tool order, and evidence handling. The model also frequently used `X-ray` where this exact-match API expects `X-RAY DIFFRACTION`; this can silently produce an empty candidate set. An empty result after a correct lookup differs from an invalid lookup, but many original-model answers treated both as proof of no match.

The SFT demonstrations explicitly teach identity lookup → filtered structure search → inspection, with no-match answers retaining the resolved accession. This makes a large SFT improvement plausible without demonstrating new biological knowledge. All evaluation evidence is present in the tools; the task tests retrieval and API composition.

Limitations of this benchmark: only three question templates; unambiguous protein queries; direct filtering by method and resolution; no-match examples all at the minimum available resolution; one model seed; no protein-family split. These choices make the pilot convenient but potentially too easy after SFT. A saturated SFT baseline would require a separately versioned, more demanding benchmark to study RL gains; changing the current held-out questions would invalidate the comparison.

## SFT sampled-probe failures (readiness probe, W&B run `b34258627f53`)

The pre-RL readiness probe sampled four attempts at temperature 0.8 for 30 training questions: 113/120 succeeded, 4 questions had mixed outcomes, and every one of the 7 failures followed one pattern:

1. The question named the protein by a gene alias (`NKAT-8`, `ZNF524`, and similar).
2. The sampled policy passed that alias straight to `search_structures` as if it were an accession and received `not_found`.
3. It then recovered the accession correctly through `search_proteins`, but next passed the UniProt accession to `get_structure` as a PDB ID, received `not_found` again, and answered `no_match`.

The greedy policy never makes this mistake on the 140 validation questions, so greedy validation is saturated at 100% while the sampled policy still fails about 6% of attempts. That gap is the only learning signal GRPO has in this benchmark; the outcome to watch is the post-training probe on the same 30 questions and seed, not the greedy validation score.

| Question type | Failure label | First tool called | Count |
|---|---|---|---:|
| alias | incorrect_no_match | search_structures | 4 |
| alias | wrong_protein | search_structures | 1 |
| standard | incorrect_no_match | search_structures | 2 |

## After GRPO: the same probe, same seed (W&B run `b34258627f53`)

Across the 100 optimizer updates, 400 training rollouts succeeded 98% of the time; 5 of the 100 rollout groups had mixed rewards and none were all-failure, so most updates carried only the KL term. Maximum recorded KL from the SFT reference was 0.0022 and policy entropy fell from 0.0079 to 0.0010: the policy sharpened rather than moved.

Repeating the 30-question, four-sample probe after training gave 119/120 (99.2%) against 113/120 (94.2%) before. All four questions that had been sampled-flaky before RL (`P43632-04`, `Q96C55-04`, `P0ADR8-01`, `Q6W0C5-04`) succeeded 4/4 afterwards; the alias-as-accession detour no longer appeared. One new failure appeared on `P76272-09`, a true no-match question: the policy resolved the protein and correctly found an empty filtered result, then emitted `status: no_match` with the accession copied into `pdb_id` and the threshold copied into `resolution_angstrom`. The evaluator rejects that answer because no-match answers must carry null structure fields.

This probe is a sampled-decoding diagnostic on training questions, so it is evidence that RL optimized its objective, not a held-out result. The held-out comparison lives in the test-split evaluations of the three checkpoints.


## Provenance and limitations

Only one training seed was used. Intervals resample proteins, not paraphrases; intervals at 0% or 100% can collapse and do not prove perfect population performance.
This pilot uses a small frozen corpus, three prompt templates, unambiguous identity queries, and server-side structure filtering. It does not establish biological reasoning or protein-family generalization.
The original validation run predates a precision-text correction affecting two training questions only; all original validation question texts were verified unchanged.
- sft checkpoint: `ExplainThisPruner/protein-structure-detective/protein-detective-v1-sft-final:v0`, digest `da74d81fb9984cc7758f281e760a46e3`.
- grpo checkpoint: `ExplainThisPruner/protein-structure-detective/protein-detective-v1-grpo-best:v3`, digest `ea7e30889178118f5ec5ec5e25a573b6`.
