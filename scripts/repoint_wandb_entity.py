"""Repoint local artifact references after runs were copied to another W&B entity.

Every reference is checked against the destination first: the artifact must
exist there under the same name and version with the same content digest.
Nothing is rewritten unless every reference verifies. Usage:
  python scripts/repoint_wandb_entity.py ExplainThisPruner davidettel-personal-org
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import wandb
from protein_detective.core import read_json, write_json

old, new = sys.argv[1], sys.argv[2]
# The destination may belong to a different W&B account: read it with WANDB_DST_API_KEY when set.
import os
api = wandb.Api(api_key=os.environ.get("WANDB_DST_API_KEY") or None, timeout=120)
print("checking destination as:", api.viewer.username)
files = [*Path('data/v1').glob('*-artifact.json'), *Path('runs').rglob('artifact.json'), *Path('runs').rglob('*.uploaded.json')]
planned, problems = [], []
for path in files:
    ref = read_json(path)
    name = ref.get('name', '')
    if not name.startswith(old + '/'):
        continue
    target = new + name[len(old):]
    try:
        remote = api.artifact(target)
        if remote.digest != ref['digest']:
            problems.append(f'{path}: digest differs at {target} ({remote.digest} vs {ref["digest"]})')
            continue
    except Exception as error:
        problems.append(f'{path}: {target} not found ({type(error).__name__})')
        continue
    planned.append((path, ref, target))
print(json.dumps({'references': len(planned) + len(problems), 'verified': len(planned), 'problems': problems[:20]}, indent=2))
if problems:
    raise SystemExit('Not rewriting: some references could not be verified in the destination entity')
for path, ref, target in planned:
    write_json(path, {**ref, 'name': target, 'previous_name': ref['name']})
print(f'Rewrote {len(planned)} references from {old} to {new}')
