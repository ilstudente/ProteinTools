import pytest
from protein_detective.budget import reservation,load_ledger
from protein_detective.core import write_json

def test_reservations_include_uncertain_and_prior_experiments(tmp_path):
    c=dict(budget_usd=50,stage_caps={'smoke':5},hourly_rate_usd=1.6,experiment='v2')
    j,cap=reservation({'jobs':[]},c,'smoke')
    assert j['reserved_usd']==1.75 and j['timeout_seconds']==3600
    write_json(tmp_path/'v1'/'budget.json',{'jobs':[j]})
    write_json(tmp_path/'budget.json',{'jobs':[j]})
    ledger=load_ledger(tmp_path)
    assert len(ledger['jobs'])==1
    _,cap=reservation(ledger,c,'smoke')
    assert cap==3.25
    j['reserved_usd']=5
    with pytest.raises(RuntimeError):reservation({'jobs':[j]},c,'smoke')


def test_grpo_timeout_remains_within_stage_and_project_caps():
    c=dict(budget_usd=50,stage_caps={'grpo':25},hourly_rate_usd=1.6,experiment='v1',grpo_job_seconds=7200)
    job,cap=reservation({'jobs':[]},c,'grpo')
    assert job['timeout_seconds']==7200 and job['reserved_usd']==3.35
    prior=dict(job,actual_estimate_usd=24)
    shorter,cap=reservation({'jobs':[prior]},c,'grpo')
    assert shorter['timeout_seconds']<7200 and shorter['reserved_usd']<=1
