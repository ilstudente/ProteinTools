from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import Snapshot, read_json, write_json
from .tracking import Tracker, preflight


def main():
    parser = argparse.ArgumentParser(description="Frozen protein tool-use experiment")
    parser.add_argument("command", choices=["preflight", "collect", "prepare", "verify", "train", "evaluate", "report", "status", "check"])
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data", default="data/v1")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--stage", choices=["smoke", "baseline", "sft", "grpo"], default="smoke")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--offline", action="store_true", help="W&B offline mode for local tests; must sync before completion")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.offline:
        config["wandb_mode"] = "offline"
    if args.command == "check":
        import subprocess,sys,xml.etree.ElementTree as ET
        with Tracker(args.runs,config,"smoke",args.run_name or "local-checks") as tracker:
            path=tracker.root/"pytest.xml"
            completed=subprocess.run([sys.executable,"-m","pytest","-q",f"--junitxml={path}"],text=True,capture_output=True)
            print(completed.stdout)
            (tracker.root/"pytest-output.txt").write_text(completed.stdout+completed.stderr)
            tracker.artifact(path,f"tests-{tracker.run_id}","verification")
            tracker.artifact(tracker.root/"pytest-output.txt",f"test-output-{tracker.run_id}","verification")
            suite=ET.parse(path).getroot().find("testsuite")
            tracker.log({f"tests/{k}":float(suite.attrib[k]) for k in ["tests","failures","errors","skipped","time"]})
            if completed.returncode:raise RuntimeError("Local verification failed")
        return
    if args.command == "status":
        import wandb
        api=wandb.Api(timeout=30)
        entity=config.get("entity") or api.default_entity
        for run in api.runs(f"{entity}/{config['project']}", filters={"group":config["experiment"]}):
            if args.run_name and run.name != args.run_name:
                continue
            print(json.dumps({"name":run.name,"state":run.state,"url":run.url,
                "episode":run.summary.get("episode"),"success":run.summary.get("validation/success"),
                "last_reward":run.summary.get("episodes/reward"),
                "readiness_mixed":run.summary.get("learning/readiness_mixed"),
                "step":run.summary.get("train/global_step"),"failure":run.summary.get("failure")}))
        return
    if args.command == "preflight":
        print(preflight(config))
        return
    if args.command == "collect":
        from .data import collect
        with Tracker(args.runs, config, "data", "collect") as tracker:
            manifest = collect(args.data, args.count, config["seed"])
            tracker.log({f"data/{k}": v for k, v in manifest.items() if isinstance(v, (int, float))})
            ref = tracker.artifact(args.data, f"{config['experiment']}-snapshot", "dataset")
            write_json(Path(args.data) / "snapshot-artifact.json", ref)
        return
    if args.command in ["prepare", "verify"]:
        from .data import generate_questions, digest
        from .episodes import scripted_episode
        with Tracker(args.runs, config, "data", args.run_name or args.command) as tracker:
            data = Path(args.data)
            if args.command == "prepare":
                questions = generate_questions(data, config["seed"])
            else:
                from .data import verify_dataset
                verify_dataset(data)
                questions = read_json(data / "questions.json")
            snap = Snapshot.load(data / "snapshot.json")
            demos = []
            for q in questions:
                r = scripted_episode(snap, q)
                assert r["reward"] == 1, (q["id"], r["failure"])
                tracker.episode(r)
                if q["split"] == "train":
                    demos.append(r)
            import random
            random.Random(config["seed"]).shuffle(demos)
            write_json(data / "demonstrations.json", demos[:300])
            tracker.log({"data/questions": len(questions), "data/solver_success": 1,
                         **{f"data/{s}_questions": sum(q["split"]==s for q in questions) for s in ["train","validation","test"]},
                         **{f"data/type_{k}": sum(q["kind"]==k for q in questions) for k in ["standard","alias","accession","no_match"]}})
            manifest = read_json(data / "manifest.json")
            manifest.update(questions_hash=digest(questions), splits_hash=digest(read_json(data/"splits.json")),
                            demonstrations_hash=digest(demos[:300]))
            write_json(data / "manifest.json", manifest)
            ref = tracker.artifact(data, f"{config['experiment']}-prepared", "dataset")
            write_json(data / "prepared-artifact.json", ref)
        return
    if args.command == "train":
        from .training import train
        train(args.data, args.runs, config, args.stage, args.checkpoint, args.run_name)
    elif args.command == "evaluate":
        from .training import evaluate_checkpoint
        evaluate_checkpoint(args.data, args.runs, config, args.checkpoint, args.split)
    elif args.command == "report":
        from .evaluation import report
        print(report(args.runs, config, args.run_name or "report"))


if __name__ == "__main__":
    main()
