"""Protein-cluster uncertainty, readiness checks and reproducible reports."""
from __future__ import annotations
from collections import Counter, defaultdict
import random
from pathlib import Path
from .core import read_json, write_json


def summarize(records, seed=42):
    if not records:
        raise ValueError("No episodes to evaluate")
    by_protein = defaultdict(list)
    for r in records:
        by_protein[r["accession"]].append(r["reward"])
    rng = random.Random(seed)
    accs = sorted(by_protein)
    boot = []
    for _ in range(2000):
        values = [v for a in rng.choices(accs, k=len(accs)) for v in by_protein[a]]
        boot.append(sum(values)/len(values))
    boot.sort()
    calls = sum(r.get("tool_calls", 0) for r in records)
    errors = sum(r.get("tool_errors", 0) for r in records)
    result = {"success": sum(r["reward"] for r in records)/len(records), "episodes": len(records),
              "proteins": len(accs), "ci_low": boot[49], "ci_high": boot[1949],
              "valid_call_rate": (calls-errors)/calls if calls else None,
              "calls_per_episode": calls/len(records),
              "tokens_per_episode": sum(r.get("generated_tokens", 0) for r in records)/len(records),
              "failures": dict(Counter(r["failure"] for r in records if r.get("failure")))}
    for kind in sorted({r["kind"] for r in records}):
        group = [r for r in records if r["kind"] == kind]
        result[f"success_{kind}"] = sum(r["reward"] for r in group)/len(group)
    positives = [r for r in records if r["kind"] != "no_match"]
    result["success_positive"] = sum(r["reward"] for r in positives)/len(positives) if positives else None
    return result



def paired_gain(before,after,seed=42):
    left={r["question_id"]:r for r in before}
    right={r["question_id"]:r for r in after}
    if len(left)!=len(before) or len(right)!=len(after) or left.keys()!=right.keys():
        raise ValueError("Comparison requires exactly the same unique questions")
    groups=defaultdict(list)
    for key,l in left.items():
        r=right[key]
        if l["question"]!=r["question"] or l["accession"]!=r["accession"]:
            raise ValueError("Held-out question changed between checkpoints")
        groups[l["accession"]].append(r["reward"]-l["reward"])
    values=[v for group in groups.values() for v in group]
    rng=random.Random(seed)
    keys=sorted(groups)
    boot=[]
    for _ in range(2000):
        sample=[v for key in rng.choices(keys,k=len(keys)) for v in groups[key]]
        boot.append(sum(sample)/len(sample))
    boot.sort()
    return {"gain":sum(values)/len(values),"ci_low":boot[49],"ci_high":boot[1949]}

def evaluate(model, runner, questions, tracker, checkpoint, step=0):
    import gzip
    # Reuse only deterministic episodes from this run and exact checkpoint.
    saved = {}
    for path in sorted(tracker.root.glob("episodes-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                r = __import__('json').loads(line)
                if r.get("checkpoint") == checkpoint and r.get("generation", {}).get("sample") is False:
                    saved[r["question_id"]] = r
    for r in getattr(tracker, "pending", []):
        if r.get("checkpoint") == checkpoint and r.get("generation", {}).get("sample") is False:
            saved[r["question_id"]] = r
    records = []
    for q in questions:
        if q["id"] in saved:
            r = saved[q["id"]]
            if r["question"] != q["text"] or r["accession"] != q["accession"]:
                raise ValueError("Resumed evaluation question changed")
            records.append(r)
            continue
        r = runner.run(model, q, sample=False)
        r["checkpoint"] = checkpoint
        tracker.episode(r)
        records.append(r)
    result = summarize(records, runner.config["seed"])
    prefix="test" if questions[0]["split"]=="test" else "validation"
    tracker.log({"checkpoint_step": step, **{f"{prefix}/{k}": v for k,v in result.items() if isinstance(v,(int,float))}})
    tracker.flush()
    return result, records


def readiness(model, runner, questions, tracker, label="sft-readiness", prefix="readiness"):
    questions = list(questions)
    random.Random(runner.config["seed"]).shuffle(questions)
    groups = []
    for q in questions[:30]:
        group = []
        for _ in range(4):
            r = runner.run(model, q, sample=True)
            tracker.episode({**r, "checkpoint": label})
            group.append(r["reward"])
        groups.append(group)
    summary = {"mixed": sum(0 < sum(g) < len(g) for g in groups)/len(groups),
               "all_zero": sum(sum(g)==0 for g in groups)/len(groups),
               "all_one": sum(sum(g)==len(g) for g in groups)/len(groups),
               "questions": len(groups), "episodes": sum(len(g) for g in groups),
               "success": sum(sum(g) for g in groups)/sum(len(g) for g in groups)}
    tracker.log({f"learning/{prefix}_{k}":v for k,v in summary.items()})
    tracker.flush()
    return summary



def grpo_training_section(root):
    """Report lines for the RL run itself: sampled probe before/after and validation over training."""
    root = Path(root)
    lines = []
    before, after = root/"grpo"/"readiness.json", root/"grpo"/"readiness-after.json"
    if before.exists():
        b = read_json(before)
        lines += ["", "## Sampled training-question probe (the objective RL optimizes)", "",
                  "Thirty fixed training questions, four samples each at temperature 0.8, identical seed before and after GRPO.", "",
                  "| Policy | Sampled success | Mixed groups | All-success groups | All-failure groups |", "|---|---:|---:|---:|---:|",
                  f"| SFT (before RL) | {b['success']:.1%} | {b['mixed']:.1%} | {b['all_one']:.1%} | {b['all_zero']:.1%} |"]
        if after.exists():
            a = read_json(after)
            lines.append(f"| SFT + GRPO (after RL) | {a['success']:.1%} | {a['mixed']:.1%} | {a['all_one']:.1%} | {a['all_zero']:.1%} |")
            lines += ["", f"Sampled success changed by {(a['success']-b['success'])*100:.1f} percentage points on these training questions. "
                      "This is a training-set, sampled-decoding diagnostic, not a held-out result."]
    steps = sorted(((int(p.stem.split("-")[1]), read_json(p)) for p in (root/"grpo").glob("validation-*.json")
                    if p.stem.split("-")[1].isdigit()), key=lambda x: x[0])
    final = root/"grpo"/"validation-summary.json"
    if steps or final.exists():
        lines += ["", "## Validation during GRPO (greedy, 140 held-out questions)", "", "| Optimizer step | Success | Valid calls | Calls/episode |", "|---:|---:|---:|---:|"]
        for step, v in steps:
            lines.append(f"| {step} | {v['success']:.1%} | {v['valid_call_rate'] if v['valid_call_rate'] is not None else 'N/A'} | {v['calls_per_episode']:.2f} |")
        if final.exists():
            v = read_json(final)
            lines.append(f"| final | {v['success']:.1%} | {v['valid_call_rate'] if v['valid_call_rate'] is not None else 'N/A'} | {v['calls_per_episode']:.2f} |")
        lines += ["", "The RL checkpoint was selected on validation success only; ties were broken toward the later checkpoint."]
    return lines


def report(root, config, run_name="report"):
    from .tracking import Tracker
    root = Path(root)/config["experiment"]
    files = sorted(root.glob("evaluation-*/summary.json"))
    summaries = [{"run":p.parent.name, **read_json(p)} for p in files]
    sft_validation = root/"sft"/"validation-summary.json"
    if sft_validation.exists():
        summaries.append({"run":"sft", "split":"validation", "label":"sft", **read_json(sft_validation)})
    content = ["# Protein Structure Detective results", "", "Single-seed pilot; uncertainty is clustered by protein.", "",
               "| Checkpoint | Split | Success | 95% interval | Episodes |", "|---|---|---:|---|---:|"]
    for s in summaries:
        content.append(f"| {s['run']} | {s['split']} | {s['success']:.1%} | {s['ci_low']:.1%}–{s['ci_high']:.1%} | {s['episodes']} |")
    tests = {s.get("label"): s for s in summaries if s["split"]=="test"}
    if {"original", "sft"} <= tests.keys():
        sft_gain = paired_gain(read_json(root/tests["original"]["run"]/"records.json"),
                               read_json(root/tests["sft"]["run"]/"records.json"), config["seed"])
        write_json(root/"sft-comparison.json", sft_gain)
        content += ["", f"SFT improvement over original: {sft_gain['gain']*100:.1f} percentage points "
                    f"(paired protein-bootstrap interval {sft_gain['ci_low']*100:.1f} to {sft_gain['ci_high']*100:.1f})."]
    if {"sft", "grpo"} <= tests.keys():
        comparison=paired_gain(read_json(root/tests["sft"]["run"]/"records.json"),
                               read_json(root/tests["grpo"]["run"]/"records.json"),config["seed"])
        write_json(root/"paired-comparison.json",comparison)
        gain = comparison["gain"]
        content += ["", f"Observed RL improvement: {gain*100:.1f} percentage points.",
                    f"Paired protein-bootstrap 95% interval: {comparison['ci_low']*100:.1f} to {comparison['ci_high']*100:.1f} points.",
                    "Target met." if gain >= .1 else "The 10-point target was not met."]
    elif (root/"grpo"/"readiness.json").exists() and read_json(root/"grpo"/"readiness.json")["mixed"] == 0:
        signal = read_json(root/"grpo"/"readiness.json")
        content += ["", "GRPO stopped at the required learning-signal gate: no sampled question had mixed rewards.",
                    f"Readiness results: {signal.get('questions',30)} questions, {signal.get('episodes',120)} trajectories; "
                    f"all-success groups {signal['all_one']:.1%}, all-failure groups {signal['all_zero']:.1%}.",
                    "No RL-trained checkpoint or RL improvement is claimed. The three-checkpoint comparison was not performed."]
        if not {"original", "sft"} <= tests.keys():
            content.append("Original and SFT test evaluation remains incomplete.")
    else:
        content += ["", "Experiment incomplete: a matched SFT and GRPO test comparison is not yet available."]
    content += grpo_training_section(root)
    failures = []
    # Held-out test failures first, trained checkpoints before the original model, then validation.
    ordered = sorted(root.glob("evaluation-*/records.json"),
                     key=lambda p: ("test" not in p.parent.name, "original" in p.parent.name, p.parent.name))
    for p in ordered:
        failures.extend({**r, "checkpoint_run": p.parent.name} for r in read_json(p) if r["reward"]==0)
    content += ["", "## Tool use and task breakdown", "",
                "| Checkpoint | No-match accuracy | Valid calls | Calls/episode | Tokens/episode |",
                "|---|---:|---:|---:|---:|"]
    for s in summaries:
        valid=s.get("valid_call_rate")
        content.append(f"| {s['run']} | {s.get('success_no_match',0):.1%} | {valid if valid is not None else 'N/A'} | {s['calls_per_episode']:.2f} | {s['tokens_per_episode']:.1f} |")
    budget_file=root.parent/"budget.json"
    if budget_file.exists():
        from .budget import charged
        ledger=read_json(budget_file)
        estimate=sum(charged(job) for job in ledger["jobs"])
        content += ["", "## Compute spend", "", f"Conservative recorded estimate/reservations: ${estimate:.2f}. This is not confirmed provider billing.",
                    f"Recorded jobs: {len(ledger['jobs'])}. Rate assumptions and per-job durations are preserved in budget.json."]
    content += ["", "## Failure inspection", ""]
    content.append(f"{len(failures)} failed held-out episodes across all evaluated checkpoints; the first 20 are listed, trained checkpoints first.")
    content.append("")
    for r in failures[:20]:
        content.append(f"- {r['checkpoint_run']} / {r['question_id']} ({r['kind']}): {r['failure']}; answer: `{r.get('answer')}`")
    review = Path("docs/failure-analysis.md")
    if review.exists():
        content += ["", review.read_text()]
    content += ["", "## Provenance and limitations", "",
                "Only one training seed was used. Intervals resample proteins, not paraphrases; intervals at 0% or 100% can collapse and do not prove perfect population performance.",
                "This pilot uses a small frozen corpus, three prompt templates, unambiguous identity queries, and server-side structure filtering. It does not establish biological reasoning or protein-family generalization.",
                "The original validation run predates a precision-text correction affecting two training questions only; all original validation question texts were verified unchanged."]
    for checkpoint in [root/"sft"/"final", root/"grpo"/"best"]:
        if (checkpoint/"artifact.json").exists():
            ref=read_json(checkpoint/"artifact.json")
            content.append(f"- {checkpoint.parent.name} checkpoint: `{ref['name']}`, digest `{ref['digest']}`.")
    path = root/"report.md"
    path.write_text("\n".join(content)+"\n")
    with Tracker(root.parent, config, "evaluation", run_name) as tracker:
        tracker.artifact(path, f"{config['experiment']}-report", "report")
        if budget_file.exists():
            tracker.artifact(budget_file, f"{config['experiment']}-budget", "budget")
        if review.exists():
            tracker.artifact(review, f"{config['experiment']}-failure-review", "evaluation")
        import wandb
        table = wandb.Table(columns=["checkpoint", "split", "success", "ci_low", "ci_high"])
        for s in summaries:
            table.add_data(s["run"],s["split"],s["success"],s["ci_low"],s["ci_high"])
        tracker.run.log({"comparison":table})
    return str(path)
