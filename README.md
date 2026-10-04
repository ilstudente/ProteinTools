# Protein Structure Detective

A small, reproducible tool-use experiment: resolve a protein in UniProt, find an experimental X-ray structure in a frozen PDB snapshot, and check its resolution. Compare Qwen3-0.6B before training, after SFT, and after GRPO. Every pipeline stage is tracked in Weights & Biases.

## Setup

```bash
uv sync --extra train --extra dev --extra dashboard
source .venv/bin/activate
wandb login
modal setup
python detective.py preflight
```

Python 3.12 is recommended. The Modal image pins the same major training packages as the local environment. Model revision is resolved to an immutable Hugging Face commit and recorded with each checkpoint; `scripts/verify_tokenizer.py` also pins it in the configuration before dispatch.

Default W&B project: `protein-structure-detective` under the personal entity `davidettel-personal` (the run entity of the `davidettel-personal-org` organization), set explicitly as `entity` in `configs/default.json` so runs never fall back to an API key's default team entity. Future Modal dispatches read the key from `WANDB_API_KEY` or `~/.netrc`, so log in with the personal org's key (`wandb login --relogin`) before dispatching. Secrets are passed through an ephemeral Modal Secret; no credentials are written into source or artifacts.

## Data and verification

```bash
python detective.py collect --count 100
python detective.py prepare
python scripts/verify_tokenizer.py
python detective.py check
```

Collection is resumable and caches raw responses in `data/v1/raw`. Each protein's PDB search follows all pages. The protein selection is a bounded reproducible sample, not a full UniProt census. The frozen snapshot—not the live public databases—defines truth.

`prepare` constructs connected-component accession splits so no PDB entry crosses splits, generates ten questions per usable protein, runs the scripted solver, and builds 300 training demonstrations. Proteins without unambiguous exact-organism mappings and numeric resolution remain searchable but do not generate questions. Actual dataset counts and type distributions are logged; unusable records can reduce the approximate 1,000-question target.

Tools use exact taxonomy IDs to avoid silently equating different strains. Fragments and mutants are accepted when mapped unambiguously to the requested accession. All split records remain searchable: held-out questions and demonstrations, not the searchable evidence, are withheld.

## Modal training

Run serially from this directory:

```bash
modal run --detach modal_app.py --stage smoke
modal run --detach modal_app.py --stage baseline
modal run --detach modal_app.py --stage sft
modal run --detach modal_app.py --stage grpo \
  --checkpoint /experiment/runs/protein-detective-v1/sft/final
```

The smoke stage performs one SFT update, saves and reloads the adapter, performs one actual GRPO update, and evaluates four validation episodes. Inspect its W&B run before proceeding.

SFT must reach 90% valid tool calls before GRPO. GRPO runs a 30-question, four-sample readiness probe and stops if no question has mixed rewards. After training it repeats the same probe with the same seed, so the sampled success rate RL actually optimizes is compared before and after on identical questions; greedy validation can saturate at 100% and hide that change. If either check fails, inspect logged traces; do not bypass it. One corrected SFT run is allowed. To preserve provenance, use a new experiment ID for a correction or the 1.7B fallback and repeat its baseline.

The dataset's filtered-search tool makes this an intentionally bounded and scriptable task. A high SFT baseline may leave little room for RL improvement. The report must retain that result rather than claim an improvement.

### Budget and recovery

The dispatcher acquires a local exclusive lock, reserves cost in `runs/budget.json (shared across model fallbacks and experiment IDs)`, and enforces one-hour smoke/SFT/evaluation timeouts and a three-hour GRPO ceiling (100 updates, a readiness probe before and after training, and validation every 25 updates). Rate estimates include an allowance above A10G pricing for CPU and RAM. Confirm current rates in `configs/default.json` before dispatch. A failed or uncertain dispatch retains its full reservation. These are conservative estimates, not a provider-enforced account spending limit or confirmed bills.

Default per-stage budgets are $5 smoke, $10 SFT, $25 GRPO, $10 baseline/evaluation. Image construction and storage incur additional provider costs; the dispatcher includes a small startup allowance, not a billing guarantee. Monitor the Modal billing dashboard as well.

Retries do not automatically rerun training. Interrupted stages retain the W&B run ID and use the latest Trainer checkpoint when resumed. Completed stages refuse accidental overwrite. Use a new experiment ID for a deliberate repetition.

A failed GRPO attempt keeps its W&B run, its archived readiness probe, and its trajectory shards; re-dispatching the same command resumes that run and reuses the probe instead of sampling 120 new episodes. To repeat a stage under a new name without touching a completed run, pass `--run-name`, for example `modal run --detach modal_app.py --stage smoke --run-name smoke-reference-fix`.

The local checkout is not the system of record. Every tracked run uploads a `source-code` artifact with `src`, `configs`, `tests`, `scripts`, `docs`, and the root entrypoints, so a lost working directory can be restored from the latest artifact, and the frozen dataset and run outputs can be fetched from the Modal Volume.

Detached dispatch keeps the current GPU call alive if the client disconnects. Check the Modal app status before resuming; the local lock does not protect against dispatch from a second machine. When SFT has saved its final adapter, resumption only completes validation. Deterministic evaluation reuses recorded episodes for the same checkpoint and question instead of repeating completed cases.

Each new dispatch saves its Modal function-call ID in the budget ledger. `python scripts/job_status.py` reads that call without starting work, and `python scripts/job_status.py --finalize` records a finished call's measured cost in the ledger when the local client died before the job ended. The launcher checks unfinished recorded calls before reserving another job; uncertain costs remain conservatively charged.

Files live on the `protein-structure-detective` Modal Volume under `/data`, `/runs`, and `/hf`. Fetch results with:

```bash
modal volume get protein-structure-detective /runs ./downloaded-runs
```

## Final evaluation

Select the GRPO adapter using validation success only; when several checkpoints tie on validation, the later one is kept because validation cannot separate them. Once selected, run each checkpoint on the test split exactly once:

```bash
modal run --detach modal_app.py --stage evaluation --split test
modal run --detach modal_app.py --stage evaluation --split test \
  --checkpoint /experiment/runs/protein-detective-v1/sft/final
modal run --detach modal_app.py --stage evaluation --split test \
  --checkpoint /experiment/runs/protein-detective-v1/grpo/best
```

Fetch the run outputs, then generate the comparison report with `python detective.py report --runs <directory-containing-protein-detective-v1>`.

Evaluation uses identical prompts, greedy decoding, six-call limits, and 4,096-token contexts for all checkpoints. The report includes protein-cluster bootstrap intervals, per-type success, no-match accuracy, tool validity, calls, and token counts. This is a single-seed pilot, not evidence of generalization to unseen protein families.

## Results (experiment `protein-detective-v1`, one seed)

| Checkpoint | Validation (140 q, greedy) | Test (100 q, greedy) | Protein-bootstrap 95% | Valid tool calls |
|---|---:|---:|---|---:|
| Original Qwen3-0.6B | 0.0% | 0.0% | 0–0% | 39% |
| SFT (LoRA, 300 demonstrations, 1 epoch) | 100.0% | 98.0% | 95–100% | 99.3% |
| SFT + GRPO (100 updates, best on validation) | 100.0% | 98.0% | 95–100% | 99.3% |

- **SFT over original:** +98.0 points on test (paired protein bootstrap 95 to 100). Both test failures are query-construction errors: the model folded the organism into the search string or reordered the name, got an empty result, then invented an accession instead of searching again.
- **GRPO over SFT:** +0.0 points on test (paired interval 0 to 0); the same two questions fail under both checkpoints. The 10-point target was not met, and the benchmark is the reason: greedy validation was already saturated at 100% before RL began.
- **What RL did change:** on the fixed 30-question, four-sample training probe, sampled success rose from 94.2% to 99.2% and mixed-reward groups fell from 13.3% to 3.3%. Over training only 5 of 100 rollout groups had mixed rewards, maximum KL from the SFT reference was 0.0022, and entropy fell from 0.0079 to 0.0010: GRPO sharpened the SFT policy and removed the sampled-only alias-as-accession detour, while introducing one malformed no-match answer.
- **Cost:** 10 Modal jobs, conservative estimate $8.26 of the $50 budget (including two full reservations for interrupted jobs). GRPO itself ran 2.1 hours on one A10G for about $3.53.

Full numbers, the failure review, and checkpoint digests are in `runs/protein-detective-v1/report.md` after `python detective.py report`; the W&B workspace URL is stored in `runs/protein-detective-v1/workspace.json`. A more demanding, separately versioned benchmark (ambiguous names, paraphrase templates, no server-side resolution filter, distractor organisms) is the prerequisite for a meaningful RL-versus-SFT comparison; changing the current held-out questions would invalidate this one.

## Moving the experiment to the personal W&B org

Experiment `protein-detective-v1` was logged to the `ExplainThisPruner` team entity because that is the default entity of the API key in `~/.netrc`, and that key cannot write to `davidettel-personal-org` (writes return 403, and the org is not among the key's organizations). Copying therefore needs the personal org's own API key. Nothing is deleted at the source; the copy is additive and resumable.

1. In the personal org, create an API key (W&B user settings) and export it only for this shell session as `WANDB_DST_API_KEY`. Keep the existing login as the source (or export `WANDB_SRC_API_KEY`).
2. Find the destination entity name the personal key can write to:

```bash
python scripts/copy_wandb_project.py --list-destinations
```

3. Copy every run (same IDs, names, groups, config, history on the original step axes, tables, summaries) and every artifact (same collection names and creation order, digests checked):

```bash
python scripts/copy_wandb_project.py ExplainThisPruner/protein-structure-detective davidettel-personal/protein-structure-detective
```

4. Repoint the local lineage references only after every artifact verifies in the destination by name, version and digest (the script reads the destination with `WANDB_DST_API_KEY`), then make the personal key the default for the workspace and audit steps with `wandb login --relogin` or `export WANDB_API_KEY=$WANDB_DST_API_KEY`, and run:

```bash
python scripts/repoint_wandb_entity.py ExplainThisPruner davidettel-personal
python scripts/create_dashboard.py
python scripts/audit_results.py
```

`configs/default.json` names `davidettel-personal` (the entity reported by `--list-destinations` for the personal key) so no future run can fall back to the team; every Modal dispatch's preflight refuses to start paid work until the logged-in key can write to that entity. Deleting the copied-from runs in `ExplainThisPruner` is a manual step in the W&B UI once the copy has been checked.

## Tracking contract

Every stage has a W&B run, grouped by experiment ID. Trainer-native metrics and custom episode metrics use separate axes. Run `python scripts/create_dashboard.py` to create or update a dedicated W&B workspace with success, learning-signal, tool-use, trace, resource, and budget sections. The final report adds a `comparison` table.

Artifacts contain source snapshots, raw/normalized data, splits, demonstrations, checkpoints, evaluation results, and **every episode** in compressed JSONL shards. Traces include complete messages, calls, responses, rewards, failure reasons, token IDs/masks, timing, generation settings, and checkpoint identity. Gold labels stay outside model input. Data solver traces have no generated-token count because no language model ran them.

An fsynced local journal precedes network uploads. W&B buffers, journals and checkpoints remain on the Modal Volume for recovery. Use `wandb sync <offline-run-directory>` after an offline run or network interruption, and verify uploads before marking the experiment complete. `--offline` is available for local development, not as a substitute for the requested online logging.

## Implementation notes

- `core.py`: pure snapshot tools, strict answer schema, entity-aware correctness.
- `data.py`: cached HTTP collection, splits, question generation.
- `episodes.py`: shared protocol, bounded inference, explicit assistant-only SFT labels.
- `training.py`: TRL SFT and GRPO, adapter lineage, validation selection.
- `tracking.py`: W&B lifecycle, artifact lineage, trace recovery.
- `evaluation.py`: clustered uncertainty and comparison report.

TRL 0.28 routes its public custom rollout function through vLLM. This project instead overrides its single generation hook for Transformers multi-turn episodes and returns an `env_mask`; native TRL continues to handle GRPO rewards, losses, reference KL and optimization. The override is intentionally version-pinned and exercised by the GPU smoke run. A copy of the SFT adapter is used as the frozen reference, not the untouched base model. TRL's own copy rounds the fp32 trainable adapter into a bf16 reference adapter; `install_sft_reference` in `training.py` replaces those tensors with exact clones and asserts equality, dtype, and freezing before any update, and `tests/test_training.py` reproduces the rounding on a tiny model.
