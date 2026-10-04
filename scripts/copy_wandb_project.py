"""Copy runs and artifacts from one W&B entity/project to another with the public SDK.

Additive only: nothing is deleted or modified at the source. Run IDs, names,
groups, job types, tags, config, history (on the original step axes), tables,
summaries and logged artifacts are recreated at the destination; artifacts keep
their collection names and are logged in creation order so versions line up.

Credentials: the source is read with WANDB_SRC_API_KEY (or the `wandb login`
key in ~/.netrc when unset; WANDB_API_KEY is deliberately ignored for the source)
and the destination is written with WANDB_DST_API_KEY (or WANDB_API_KEY, or the
~/.netrc key). Use two keys when the destination is a different W&B
account, such as a personal org. Keys are read from the environment only and
never printed or stored by this script.

Usage:
  python scripts/copy_wandb_project.py --list-destinations
  python scripts/copy_wandb_project.py SRC_ENTITY/SRC_PROJECT DST_ENTITY/DST_PROJECT [--ids id1,id2] [--limit N] [--skip-artifacts]
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("WANDB_REPORT_API_DISABLE_MESSAGE", "True")
os.environ.setdefault("WANDB_SILENT", "true")
import wandb

def netrc_key():
    import netrc
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth[2] if auth else None
    except (FileNotFoundError, netrc.NetrcParseError):
        return None


# Source: WANDB_SRC_API_KEY, else the `wandb login` key in ~/.netrc (never WANDB_API_KEY, which may already
# hold the destination key). Destination: WANDB_DST_API_KEY, else WANDB_API_KEY, else the ~/.netrc key.
src_key = os.environ.get("WANDB_SRC_API_KEY") or netrc_key()
dst_key = os.environ.get("WANDB_DST_API_KEY") or os.environ.get("WANDB_API_KEY") or netrc_key()
if "--list-destinations" in sys.argv:
    # Show which entities the destination key can write runs to, without revealing the key.
    viewer = wandb.Api(api_key=dst_key, timeout=60).viewer
    print(json.dumps({"username": viewer.username, "default_entity": viewer.entity,
                      "teams": list(getattr(viewer, "teams", []) or [])}, indent=2))
    print("Pass one of these as DST_ENTITY; a personal account's own entity is its username.")
    raise SystemExit(0)
src_path, dst_path = sys.argv[1], sys.argv[2]
ids = None
limit = None
skip_artifacts = "--skip-artifacts" in sys.argv
for i, arg in enumerate(sys.argv):
    if arg == "--ids":
        ids = set(sys.argv[i + 1].split(","))
    if arg == "--limit":
        limit = int(sys.argv[i + 1])
dst_entity, dst_project = dst_path.split("/")
api = wandb.Api(api_key=src_key, timeout=120)
runs = sorted(api.runs(src_path), key=lambda r: r.created_at)
if ids:
    runs = [r for r in runs if r.id in ids]
if limit:
    runs = runs[:limit]
print(f"COPY {len(runs)} runs {src_path} -> {dst_path}", flush=True)
state_path = Path("runs") / "wandb-copy-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": {}, "artifacts": {}}
axes = ["episode", "optimizer_step", "checkpoint_step", "train/global_step"]
axis_of = {"episodes/": "episode", "learning/": "optimizer_step", "validation/": "checkpoint_step", "test/": "checkpoint_step"}


def fetch_source(source, tmp):
    """Read everything from the source while no destination run is open.

    Once wandb.init has opened the destination run, W&B's file downloader uses
    that run's credentials, which cannot read the source project. So history,
    table files and artifact contents are fetched here with the source API only.
    """
    history = list(source.scan_history())
    tables = {}
    for row in history:
        for value in row.values():
            if isinstance(value, dict) and value.get("_type") == "table-file" and value["path"] not in tables:
                source.file(value["path"]).download(root=tmp, replace=True, api=api)
                tables[value["path"]] = json.loads((Path(tmp) / value["path"]).read_text())
    used = [] if skip_artifacts else list(source.used_artifacts())
    logged = []
    if not skip_artifacts:
        for artifact in sorted(source.logged_artifacts(), key=lambda a: a.created_at):
            # run_table artifacts are regenerated automatically when the tables are re-logged.
            if artifact.type in {"wandb-history", "job", "run_table"} or artifact.name in state["artifacts"]:
                continue
            root = artifact.download(root=str(Path(tmp) / "artifacts" / artifact.name.replace(":", "_")))
            logged.append((artifact, root))
    return history, tables, used, logged


for source in runs:
    if source.id in state["runs"]:
        print(f"SKIP run {source.id} {source.name} (already copied)", flush=True)
        continue
    started = time.time()
    notes = f"Copied from {source.url} (original created {source.created_at}).\n" + (source.notes or "")
    tmp_holder = tempfile.TemporaryDirectory()
    tmp = tmp_holder.name
    history, tables, used_artifacts, logged_artifacts = fetch_source(source, tmp)
    run = wandb.init(entity=dst_entity, project=dst_project, id=source.id, name=source.name, group=source.group,
                     job_type=source.job_type, tags=source.tags, config=dict(source.config), notes=notes,
                     resume="allow", settings=wandb.Settings(init_timeout=120, api_key=dst_key))
    for axis in axes:
        run.define_metric(axis)
    for prefix, axis in axis_of.items():
        run.define_metric(prefix + "*", step_metric=axis)
    rows = 0
    for row in history:
        step = row.pop("_step", None)
        payload = {}
        for key, value in row.items():
            if key in {"_runtime", "_timestamp"} or value is None:
                continue
            if isinstance(value, dict) and value.get("_type") == "table-file":
                table = tables[value["path"]]
                payload[key] = wandb.Table(columns=table["columns"], data=table["data"])
            elif isinstance(value, dict) and "_type" in value:
                continue  # other media types are not used by this project
            else:
                payload[key] = value
        if payload:
            run.log(payload, step=int(step) if step is not None else None)
            rows += 1
    # Explicit summary keys that were set directly rather than through log (failure, logging_complete, ...).
    summary = {k: v for k, v in dict(source.summary).items()
               if not k.startswith("_") and not (isinstance(v, dict) and "_type" in v)}
    run.summary.update(summary)
    linked = 0
    for used in used_artifacts:
        try:
            run.use_artifact(f"{dst_path}/{used.name}")
            linked += 1
        except Exception as error:
            print(f"  WARN could not link used artifact {used.name}: {type(error).__name__}", flush=True)
    logged = 0
    for artifact, root in logged_artifacts:
        copy = wandb.Artifact(artifact.name.split(":")[0], type=artifact.type,
                              description=artifact.description, metadata=artifact.metadata)
        copy.add_dir(root)
        run.log_artifact(copy, aliases=[a for a in artifact.aliases if not a.startswith("v")])
        copy.wait()
        match = copy.digest == artifact.digest
        state["artifacts"][artifact.name] = {"destination": copy.qualified_name, "digest": copy.digest, "digest_match": match}
        logged += 1
        if not match:
            print(f"  WARN digest differs for {artifact.name}: {artifact.digest} -> {copy.digest}", flush=True)
    run.finish(exit_code=1 if source.state in {"failed", "crashed"} else 0)
    tmp_holder.cleanup()
    state["runs"][source.id] = {"name": source.name, "history_rows": rows, "artifacts": logged, "used_links": linked,
                                "destination": f"{dst_path}/{source.id}"}
    state_path.parent.mkdir(exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    print(f"RUN {source.id} {source.name}: {rows} history rows, {logged} artifacts, {linked} lineage links, {time.time()-started:.0f}s", flush=True)
print("COPY COMPLETE", flush=True)
