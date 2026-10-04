import gzip
import json
from types import SimpleNamespace

import pytest

from protein_detective.evaluation import evaluate


def test_evaluation_resume_skips_only_matching_deterministic_records(tmp_path):
    record = dict(question_id='q1', question='one', accession='A', kind='standard',
                  reward=1, checkpoint='adapter', generation={'sample': False})
    with gzip.open(tmp_path/'episodes-00000.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(record)+'\n')
    emitted = []
    runner = SimpleNamespace(config={'seed': 42}, run=lambda model, q, sample: {
        **record, 'question_id': q['id'], 'question': q['text']})
    tracker = SimpleNamespace(root=tmp_path, pending=[], episode=emitted.append,
                              log=lambda x: None, flush=lambda: None)
    questions = [dict(id='q1', text='one', accession='A', split='validation'),
                 dict(id='q2', text='two', accession='A', split='validation')]
    summary, records = evaluate(None, runner, questions, tracker, 'adapter')
    assert summary['episodes'] == 2
    assert [r['question_id'] for r in emitted] == ['q2']
    questions[0]['text'] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        evaluate(None, runner, questions, tracker, 'adapter')
    emitted.clear()
    evaluate(None, runner, questions, tracker, 'another-adapter')
    assert len(emitted) == 2



def test_grpo_training_section_reports_probe_and_validation(tmp_path):
    from protein_detective.core import write_json
    from protein_detective.evaluation import grpo_training_section
    assert grpo_training_section(tmp_path) == []
    (tmp_path/'grpo').mkdir()
    write_json(tmp_path/'grpo'/'readiness.json', dict(success=.94, mixed=.13, all_one=.87, all_zero=0))
    write_json(tmp_path/'grpo'/'readiness-after.json', dict(success=.99, mixed=.03, all_one=.97, all_zero=0))
    write_json(tmp_path/'grpo'/'validation-25.json', dict(success=1.0, valid_call_rate=1.0, calls_per_episode=2.8))
    write_json(tmp_path/'grpo'/'validation-summary.json', dict(success=.99, valid_call_rate=.98, calls_per_episode=2.9))
    text = '\n'.join(grpo_training_section(tmp_path))
    assert '| SFT (before RL) | 94.0% |' in text and '| SFT + GRPO (after RL) | 99.0% |' in text
    assert 'changed by 5.0 percentage points' in text
    assert '| 25 | 100.0% |' in text and '| final | 99.0% |' in text
