import gzip
import json
import sys
from types import SimpleNamespace
from protein_detective.tracking import Tracker

class FakeRun:
    def __init__(self):self.config={};self.summary={}
    def define_metric(self,*a,**kw):pass
    def log(self,*a,**kw):pass
    def log_artifact(self,*a,**kw):pass
    def finish(self,*a,**kw):pass
class Artifact:
    digest='abc'
    def __init__(self,*a,**kw):pass
    def add_file(self,*a,**kw):pass
    def add_dir(self,*a,**kw):pass
class Table:
    def __init__(self,*a,**kw):pass
    def add_data(self,*a,**kw):pass


def test_recovery_keeps_id_and_episodes(tmp_path,monkeypatch):
    fake=SimpleNamespace(init=lambda **kw:FakeRun(),Artifact=Artifact,Table=Table,Settings=lambda **kw:None)
    monkeypatch.setitem(sys.modules,'wandb',fake)
    config={'experiment':'test','project':'test','seed':42,'wandb_mode':'offline'}
    first=Tracker(tmp_path,config,'smoke')
    first.episode({'question':'q','reward':1,'messages':[]})
    resumed=Tracker(tmp_path,config,'smoke')
    assert resumed.run_id==first.run_id and resumed.count==1 and len(resumed.pending)==1
    resumed.episode({'question':'q2','reward':0,'messages':[]})
    resumed.finish()
    rows=[]
    for p in resumed.root.glob('episodes-*.gz'):
        with gzip.open(p,'rt') as f:rows.extend(json.loads(line) for line in f)
    assert len(rows)==2 and len({r['episode_id'] for r in rows})==2
    assert json.loads(resumed.state_path.read_text())['finished']


def test_recovery_after_archive_before_journal_clear(tmp_path,monkeypatch):
    fake=SimpleNamespace(init=lambda **kw:FakeRun(),Artifact=Artifact,Table=Table,Settings=lambda **kw:None)
    monkeypatch.setitem(sys.modules,'wandb',fake)
    config={'experiment':'test','project':'test','seed':42,'wandb_mode':'offline'}
    first=Tracker(tmp_path,config,'smoke')
    row=first.episode({'question':'q','reward':1,'messages':[]})
    # Simulate the durable archive existing but the network upload never finishing.
    path=first.root/'episodes-00000.jsonl.gz'
    with gzip.open(path,'wt') as f:f.write(json.dumps(row)+'\n')
    resumed=Tracker(tmp_path,config,'smoke')
    assert resumed.count==1 and resumed.pending==[]
    assert path.with_suffix('.uploaded.json').exists()
    resumed.episode({'question':'next','reward':0,'messages':[]})
    resumed.finish()
    ids=[]
    for p in resumed.root.glob('episodes-*.gz'):
        with gzip.open(p,'rt') as f:ids.extend(json.loads(line)['episode_id'] for line in f)
    assert len(ids)==len(set(ids))==2
