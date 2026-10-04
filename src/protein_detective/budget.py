"""Conservative per-project reservations shared by all experiment configurations."""
from pathlib import Path
import time
from .core import read_json,write_json


def load_ledger(root):
    root=Path(root)
    jobs={}
    # Migrate older per-experiment ledgers without counting the same job twice.
    for path in [*sorted(root.glob('*/budget.json')),root/'budget.json']:
        if path.exists():
            for job in read_json(path)['jobs']:
                job=dict(job)
                job.setdefault('experiment',path.parent.name)
                jobs[(job['started_at'],job['stage'])]=job
    return {'jobs':list(jobs.values())}


def charged(job):
    value=job.get('actual_estimate_usd')
    return job['reserved_usd'] if value is None else value


def reservation(ledger,config,stage):
    key='evaluation' if stage in {'baseline','evaluation'} else stage
    used=sum(charged(j) for j in ledger['jobs'])
    stage_used=sum(charged(j) for j in ledger['jobs'] if j['budget_stage']==key)
    cap=min(config['budget_usd']-used,config['stage_caps'][key]-stage_used)
    ceiling=config.get('grpo_job_seconds',7200) if stage=='grpo' else config.get('max_job_seconds',3600)
    timeout=min(ceiling,int(max(0,cap-.15)/config['hourly_rate_usd']*3600*.9))
    if timeout<180:raise RuntimeError('Insufficient remaining project budget for a safe job timeout')
    job={'stage':stage,'budget_stage':key,'experiment':config['experiment'],
         'reserved_usd':timeout/3600*config['hourly_rate_usd']+.15,'timeout_seconds':timeout,
         'rate':config['hourly_rate_usd'],'actual_estimate_usd':None,'started_at':time.time(),'status':'reserved'}
    return job,cap
