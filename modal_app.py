"""Budgeted, serial Modal dispatch. Run: modal run modal_app.py --stage smoke."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import modal

# Allow the documented direct Modal entrypoint before an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

app = modal.App("protein-structure-detective")
volume = modal.Volume.from_name("protein-structure-detective", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.2.0", "trl==0.28.0", "peft==0.18.1",
                 "datasets==4.5.0", "accelerate==1.12.0", "wandb==0.25.0", "requests==2.34.2")
    .env({"HF_HOME":"/experiment/hf", "TOKENIZERS_PARALLELISM":"false", "PYTHONPATH":"/root/project/src"})
    .add_local_dir("src", remote_path="/root/project/src", ignore=["__pycache__", "*.egg-info"])
    .add_local_dir("configs", remote_path="/root/project/configs")
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml"))


@app.cls(image=image, gpu="A10G", cpu=4, memory=16384,
         volumes={"/experiment": volume}, timeout=3600, retries=0, max_containers=1, scaledown_window=2)
class Worker:
    @modal.method()
    def execute(self, stage: str, config: dict, checkpoint: str | None = None, split: str = "validation", run_name: str | None = None):
        import time
        os.chdir("/root/project")
        from protein_detective.training import train, evaluate_checkpoint
        started = time.monotonic()
        try:
            if stage=="evaluation":
                result=evaluate_checkpoint("/experiment/data", "/experiment/runs", config, checkpoint, split)
            else:
                result=train("/experiment/data", "/experiment/runs", config, stage, checkpoint, run_name)
            return {"result":result,"seconds":time.monotonic()-started}
        except Exception as error:
            import traceback
            return {"error":str(error),"traceback":traceback.format_exc(),"seconds":time.monotonic()-started}
        finally:
            volume.commit()


@app.local_entrypoint()
def main(stage: str = "smoke", config_path: str = "configs/default.json", data: str = "data/v1",
         checkpoint: str = "", split: str = "validation", run_name: str = ""):
    import fcntl
    import netrc
    import time
    import wandb
    from protein_detective.core import read_json, write_json
    from protein_detective.tracking import preflight
    if stage not in {"smoke","baseline","sft","grpo","evaluation"}:
        raise ValueError("Unknown stage")
    config=read_json(config_path)
    from protein_detective.data import verify_dataset
    verify_dataset(data)
    # Advisory file lock covers preflight, reservation and the full synchronous invocation.
    directory=Path("runs")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"dispatch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path=directory/"budget.json"
        from protein_detective.budget import load_ledger, reservation
        ledger=load_ledger(directory)
        for previous in ledger["jobs"]:
            if previous.get("function_call_id") and previous.get("status") != "completed":
                try:
                    modal.FunctionCall.from_id(previous["function_call_id"]).get(timeout=0)
                except TimeoutError:
                    raise RuntimeError("A detached GPU call is still running; inspect scripts/job_status.py before dispatching another job")
        job,cap=reservation(ledger,config,stage)
        timeout=job["timeout_seconds"]
        entity,url=preflight(config)
        config.update(on_modal=True,entity=entity,run_cap_usd=job["reserved_usd"],job_timeout_seconds=timeout,
                      prior_estimate_usd=sum(__import__("protein_detective.budget",fromlist=["charged"]).charged(j) for j in ledger["jobs"]))
        credential=os.getenv("WANDB_API_KEY")
        if not credential:
            auth=netrc.netrc().authenticators("api.wandb.ai")
            credential=auth[2] if auth else None
        if not credential:
            raise RuntimeError("No W&B API key available for Modal Secret")
        secret=modal.Secret.from_dict({"WANDB_API_KEY":credential,"WANDB_ENTITY":entity})
        with volume.batch_upload(force=True) as upload:
            upload.put_directory(data,"/data")
        ledger["jobs"].append(job)
        write_json(ledger_path,ledger)
        print(f"Reserved ${job['reserved_usd']:.2f}; timeout {timeout}s. W&B: {url}")
        started=time.monotonic()
        try:
            # The per-call override applies the secret and budget-derived timeout.
            call=Worker.with_options(timeout=timeout,secrets=[secret])().execute.spawn(stage,config,checkpoint or None,split,run_name or None)
            job["function_call_id"] = call.object_id
            write_json(ledger_path, ledger)
            print(f"Recovery handle: {call.object_id}")
            result=call.get()
            job.update(status="completed",actual_estimate_usd=(time.monotonic()-started)/3600*config["hourly_rate_usd"]+.15,
                       result=result)
            print(json.dumps(result,indent=2))
            if "error" in result:
                raise RuntimeError(result["error"])
        except BaseException:
            # Keep the full reservation charged after uncertainty or abrupt cancellation.
            job["status"]="failed_or_interrupted"
            raise
        finally:
            write_json(ledger_path,ledger)
            with volume.batch_upload(force=True) as upload:
                upload.put_file(ledger_path,"/runs/budget.json")
