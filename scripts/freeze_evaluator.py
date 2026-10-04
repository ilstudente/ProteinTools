"""Archive evaluator-only answer sets, separately from model-visible tools."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from protein_detective.core import Snapshot, read_json, write_json, qualifying
from protein_detective.data import verify_dataset, digest
from protein_detective.tracking import Tracker

config = read_json('configs/default.json')
manifest = verify_dataset('data/v1')
snapshot = Snapshot.load('data/v1/snapshot.json')
questions = read_json('data/v1/questions.json')
answers = {q['id']: {'accession': q['accession'], 'taxon': snapshot.proteins[q['accession']]['taxon'],
                     'method': q['method'], 'threshold': q['threshold'],
                     'qualifying_structures': qualifying(snapshot, q)} for q in questions}
with Tracker('runs', config, 'data', 'evaluator-freeze') as tracker:
    path = tracker.root/'answer-sets.json'
    write_json(path, {'snapshot_hash': manifest['snapshot_hash'], 'questions_hash': manifest['questions_hash'],
                     'answers_hash': digest(answers), 'answers': answers})
    tracker.artifact(path, f"{config['experiment']}-evaluator", 'evaluator')
    tracker.log({'evaluator/questions': len(answers), 'evaluator/no_match': sum(not r['qualifying_structures'] for r in answers.values())})
