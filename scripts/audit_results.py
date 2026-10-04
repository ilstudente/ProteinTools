"""Verify local result completeness and immutable W&B artifact references."""
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import wandb
from protein_detective.core import read_json, write_json
from protein_detective.data import verify_dataset
from protein_detective.tracking import Tracker

config = read_json('configs/default.json')
root = Path('runs')/config['experiment']
verify_dataset('data/v1')
questions = {q['id']: q for q in read_json('data/v1/questions.json')}
api = wandb.Api(timeout=60)
references = {}
for path in [Path('data/v1/snapshot-artifact.json'), Path('data/v1/prepared-artifact.json'),
             *root.glob('*/final/artifact.json'), *root.glob('*/best/artifact.json'),
             *root.glob('*/episodes-*.uploaded.json')]:
    ref = read_json(path)
    if ref.get('pending_sync'):
        raise RuntimeError(f'Unsynchronized artifact: {path}')
    if ':v' not in ref['name']:
        raise RuntimeError(f'Non-versioned artifact reference: {path}')
    references[ref['name']] = ref['digest']
for name, digest in references.items():
    artifact = api.artifact(name)
    if artifact.digest != digest:
        raise RuntimeError(f'Artifact digest mismatch: {name}')

counts = {}
for directory in sorted(root.iterdir()):
    if not directory.is_dir():
        continue
    rows = [json.loads(line) for path in directory.glob('episodes-*.jsonl.gz')
            for line in gzip.open(path, 'rt')]
    ids = [r['episode_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'Duplicate episode records: {directory}')
    if rows:
        counts[directory.name] = len(rows)
    if directory.name == 'sft' and (directory/'validation-summary.json').exists():
        expected = {q['id'] for q in questions.values() if q['split'] == 'validation'}
        if {r['question_id'] for r in rows} != expected or len(rows) != len(expected):
            raise RuntimeError('Recovered SFT validation is incomplete or duplicated')
    if directory.name == 'grpo' and (directory/'readiness-after.json').exists():
        # Expected: probe before and after, 400 training rollouts, three in-loop and one final validation.
        buckets = {}
        for r in rows:
            c = r['checkpoint']
            key = ('training' if c.startswith('optimizer-step-') else 'inloop_validation' if c.startswith('step-')
                   else 'final_validation' if c.startswith('/experiment') else c)
            buckets[key] = buckets.get(key, 0) + 1
        expected = {'sft-readiness': 120, 'grpo-readiness': 120, 'training': 400, 'inloop_validation': 420, 'final_validation': 140}
        if buckets != expected:
            raise RuntimeError(f'GRPO trajectory buckets differ from expectation: {buckets}')
        counts['grpo_buckets'] = buckets
    if directory.name.startswith('evaluation-') and (directory/'summary.json').exists():
        summary = read_json(directory/'summary.json')
        records = read_json(directory/'records.json')
        expected = {q['id'] for q in questions.values() if q['split'] == summary['split']}
        if {r['question_id'] for r in records} != expected or len(records) != len(expected):
            raise RuntimeError(f'Incomplete or repeated evaluation: {directory}')
        for r in records:
            if r['question'] != questions[r['question_id']]['text']:
                raise RuntimeError(f'Question text drift: {directory}')
        if len(rows) != len(records):
            raise RuntimeError(f'Trajectory archive count differs from evaluation: {directory}')

result = {'immutable_artifacts_verified': len(references), 'unique_episode_counts': counts,
          'snapshot_and_split_hashes_verified': True, 'heldout_question_texts_verified': True}
run_name = sys.argv[sys.argv.index('--run-name') + 1] if '--run-name' in sys.argv else 'artifact-audit'
with Tracker('runs', config, 'smoke', run_name) as tracker:
    path = tracker.root/'audit.json'
    write_json(path, result)
    tracker.artifact(path, f"{config['experiment']}-artifact-audit", 'verification')
    tracker.log({'verification/artifacts': len(references), 'verification/pass': 1})
print(json.dumps(result, indent=2))
