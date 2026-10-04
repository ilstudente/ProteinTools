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
