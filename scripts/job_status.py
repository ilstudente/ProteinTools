"""Read the most recent durable Modal call without dispatching new GPU work.

With --finalize, record a completed call in the budget ledger exactly as the
dispatcher would have, for the case where the local client disconnected while a
detached GPU call kept running.
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import modal
from protein_detective.budget import load_ledger
from protein_detective.core import read_json, write_json

ledger_path = Path('runs')/'budget.json'
ledger = load_ledger('runs')
jobs = [j for j in ledger['jobs'] if j.get('function_call_id')]
if not jobs:
    raise SystemExit('No durable call handle recorded. Check the Modal app dashboard.')
job = jobs[-1]
try:
    result = modal.FunctionCall.from_id(job['function_call_id']).get(timeout=0)
except TimeoutError:
    result = {'status': 'still_running'}
print(json.dumps({'stage': job['stage'], 'function_call_id': job['function_call_id'], 'status': job.get('status'), 'result': result}, indent=2))
if '--finalize' in sys.argv and result.get('status') != 'still_running' and job.get('status') != 'completed':
    # Charge the measured container time plus the same startup allowance the dispatcher uses.
    seconds = result.get('seconds') or job['timeout_seconds']
    job.update(status='completed', actual_estimate_usd=seconds/3600*job['rate']+.15, result=result,
               finalized_at=time.time(), finalized_by='scripts/job_status.py')
    write_json(ledger_path, ledger)
    volume = modal.Volume.from_name('protein-structure-detective')
    with volume.batch_upload(force=True) as upload:
        upload.put_file(ledger_path, '/runs/budget.json')
    print(f"Ledger finalized: {job['stage']} charged ${job['actual_estimate_usd']:.2f}")
    if 'error' in result:
        raise SystemExit(f"Remote job failed: {result['error']}")
