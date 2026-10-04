"""W&B lifecycle, immutable artifacts and crash-recoverable complete traces."""
from __future__ import annotations
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path
from .core import write_json, read_json


def provenance():
    packages = {}
    for name in ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "wandb", "modal"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    return {"packages": packages, "python": platform.python_version(), "code_commit": commit,
            "modal": {k: os.environ[k] for k in ["MODAL_TASK_ID", "MODAL_FUNCTION_ID"] if k in os.environ}}


class Tracker:
    def __init__(self, root, config, job_type, suffix=None):
        import wandb
        self.config = dict(config)
        if job_type == "data" or not config.get("on_modal", False):
            self.config["hourly_rate_usd"] = 0
        self.root = Path(root) / config["experiment"] / (suffix or job_type)
        self.root.mkdir(parents=True, exist_ok=True)
        state_path = self.root / "run.json"
        if state_path.exists():
            state = read_json(state_path)
            if state.get("finished"):
                raise RuntimeError(f"Run already finished: {self.root}. Choose a new experiment/suffix.")
            self.run_id = state["id"]
        else:
            self.run_id = uuid.uuid4().hex[:12]
        self.state_path = state_path
        self.state = {"id": self.run_id, "finished": False}
        write_json(state_path, self.state)
        self.started = time.monotonic()
        self.count = 0
        self.pending = []
        self.samples = []
        self.shard = len(list(self.root.glob("episodes-*.jsonl.gz")))
        self.journal = self.root / "episodes.pending.jsonl"
        if self.journal.exists():
            for line in self.journal.read_text().splitlines():
                try:
                    self.pending.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # only an incomplete last line after abrupt termination
        self.count = int(read_json(self.root / "counter.json")["count"]) if (self.root / "counter.json").exists() else 0
        archived_ids = set()
        self.unuploaded = []
        for archived in self.root.glob("episodes-*.jsonl.gz"):
            if not archived.with_suffix(".uploaded.json").exists():
                self.unuploaded.append(archived)
            with gzip.open(archived, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    archived_ids.add(row["episode_id"])
                    self.count = max(self.count, int(row["episode_id"].rsplit(":", 1)[1]))
        self.pending = [row for row in self.pending if row["episode_id"] not in archived_ids]
        for row in self.pending:
            self.count = max(self.count, int(row["episode_id"].rsplit(":", 1)[1]))
        # Reconcile a crash between artifact creation and journal clearing.
        self.journal.write_text("".join(json.dumps(row) + "\n" for row in self.pending))
        mode = config.get("wandb_mode", "online")
        source_info = provenance()
        self.run = wandb.init(project=config["project"], entity=config.get("entity"), id=self.run_id,
                              resume="allow" if mode == "online" else None, group=config["experiment"],
                              job_type=job_type, name=suffix or job_type, dir=str(self.root), mode=mode,
                              config={**config, **source_info}, tags=[config.get("model", "data").split("/")[-1], str(config["seed"]),
                                  config["experiment"], source_info["code_commit"]],
                              settings=wandb.Settings(init_timeout=60))
        if self.run.summary.get("failure"):
            self.run.summary["recovery/previous_failure"] = self.run.summary["failure"]
            del self.run.summary["failure"]
        for axis in ["episode", "optimizer_step", "checkpoint_step"]:
            self.run.define_metric(axis)
        for prefix, axis in [("episodes/*", "episode"), ("learning/*", "optimizer_step"), ("validation/*", "checkpoint_step"), ("test/*", "checkpoint_step")]:
            self.run.define_metric(prefix, step_metric=axis)
        self.run.config.update({"run_id": self.run_id}, allow_val_change=True)
        self.log_source()
        for path in self.unuploaded:
            shard_id = path.name.split("-")[1].split(".")[0]
            ref = self.artifact(path, f"trajectories-{self.run_id}-{shard_id}", "trajectories")
            write_json(path.with_suffix(".uploaded.json"), ref)

    def log_source(self):
        import wandb
        artifact = wandb.Artifact(f"source-{self.run_id}", type="source-code")
        paths = []
        for directory in [Path("src"), Path("configs"), Path("tests"), Path("scripts"), Path("docs")]:
            if directory.exists():
                paths.extend(p for p in directory.rglob("*") if p.suffix in {".py", ".json", ".toml", ".md"} and "__pycache__" not in p.parts)
        paths.extend(Path(p) for p in ["pyproject.toml", "uv.lock", "modal_app.py", "detective.py", "README.md"] if Path(p).exists())
        for path in paths:
            artifact.add_file(str(path), name=str(path))
        if paths:
            self.run.log_artifact(artifact)

    def log(self, metrics):
        elapsed = time.monotonic() - self.started
        estimated = elapsed / 3600 * self.config.get("hourly_rate_usd", 0)
        self.run.log({**metrics, "resources/elapsed_seconds": elapsed,
                      "budget/estimated_run_usd": estimated,
                      "budget/estimated_project_usd": self.config.get("prior_estimate_usd",0) + estimated,
                      "budget/remaining_project_usd": max(0,self.config.get("budget_usd",50)-self.config.get("prior_estimate_usd",0)-estimated),
                      "budget/remaining_run_usd": max(0, self.config.get("run_cap_usd", 50) - estimated)})

    def artifact(self, path, name, kind, aliases=None):
        import wandb
        path = Path(path)
        artifact = wandb.Artifact(name, type=kind)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases or [])
        if self.config.get("wandb_mode", "online") == "online":
            artifact.wait()
            ref = {"name": artifact.qualified_name, "digest": artifact.digest}
        else:
            ref = {"name": name, "digest": artifact.digest, "pending_sync": True}
        return ref

    def episode(self, record):
        self.count += 1
        record = {**record, "episode_id": f"{self.run_id}:{self.count}", "run_id": self.run_id}
        # Write and fsync before metrics or network uploads.
        with self.journal.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        write_json(self.root / "counter.json", {"count": self.count})
        self.pending.append(record)
        if len(self.samples) < 40:
            self.samples.append(record)
        metrics={"episode":self.count,"episodes/reward":record["reward"]}
        for key in ["tool_calls","tool_errors","generated_tokens","seconds"]:
            if key in record:
                metrics[f"episodes/{key}"]=record[key]
        if "generated_tokens" in record and record.get("seconds",0)>0:
            metrics["episodes/tokens_per_second"]=record["generated_tokens"]/record["seconds"]
        self.log(metrics)
        if len(self.pending) >= 25:
            self.flush()
        return record

    def flush(self):
        if not self.pending:
            return
        path = self.root / f"episodes-{self.shard:05d}.jsonl.gz"
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as f:
            for record in self.pending:
                f.write(json.dumps(record, allow_nan=False) + "\n")
        temporary.replace(path)
        # Journals remain available if upload fails.
        ref = self.artifact(path, f"trajectories-{self.run_id}-{self.shard:05d}", "trajectories")
        write_json(path.with_suffix(".uploaded.json"), ref)
        self.journal.write_text("")
        self.pending.clear()
        self.shard += 1

    def finish(self, error=None):
        import wandb
        try:
            self.flush()
            table = wandb.Table(columns=["episode_id", "question", "reward", "failure", "trace"])
            for r in self.samples:
                table.add_data(r["episode_id"], r["question"], r["reward"], r.get("failure"), json.dumps(r.get("messages", [])))
            self.run.log({"examples": table})
            if error:
                self.run.summary["failure"] = str(error)
            elif self.run.summary.get("failure") is not None:
                del self.run.summary["failure"]
            self.run.summary["logging_complete"] = self.config.get("wandb_mode", "online") == "online"
            self.run.finish(exit_code=1 if error else 0)
            self.state.update(finished=not bool(error), failed=bool(error), error=str(error) if error else None)
            write_json(self.state_path, self.state)
        except Exception:
            self.state["needs_sync"] = True
            write_json(self.state_path, self.state)
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.finish(error)


def preflight(config):
    import wandb
    api = wandb.Api(timeout=30)
    entity = config.get("entity") or api.default_entity
    if not entity:
        raise RuntimeError("No W&B entity available; authenticate with wandb login")
    # A short CPU run verifies actual project write access.
    run = wandb.init(project=config["project"], entity=entity, job_type="smoke", group=config["experiment"],
                     name="authentication-preflight", config={"purpose": "verify project write access"})
    run.log({"authentication/ok": 1})
    url = run.url
    run.finish()
    return entity, url
