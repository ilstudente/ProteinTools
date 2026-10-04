"""Create or update the experiment's dedicated W&B workspace."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from protein_detective.core import read_json,write_json

config=read_json('configs/default.json')
entity=config.get('entity') or wandb.Api().default_entity
state=Path('runs')/config['experiment']/'workspace.json'
sections=[
 ws.Section(name='Task success',panels=[
  wr.BarPlot(title='Final test success',metrics=['test/success']),
  wr.BarPlot(title='Validation success',metrics=['validation/success']),
  wr.LinePlot(title='Validation over training',x='checkpoint_step',y=['validation/success']),
  wr.BarPlot(title='Success by task type',metrics=['validation/success_standard','validation/success_alias','validation/success_accession','validation/success_no_match'])]),
 ws.Section(name='Learning signal',panels=[
  wr.LinePlot(title='GRPO reward',x='train/global_step',y=['train/reward']),
  wr.LinePlot(title='Mixed-reward groups',x='optimizer_step',y=['learning/mixed_groups','learning/all_zero_groups','learning/all_one_groups']),
  wr.LinePlot(title='SFT loss',x='train/global_step',y=['train/loss']),
  wr.LinePlot(title='KL and entropy',x='train/global_step',y=['train/kl','train/entropy']),
  wr.BarPlot(title='Sampled probe success: before vs after RL',metrics=['learning/readiness_success','learning/readiness_after_success']),
  wr.BarPlot(title='Reference adapter verified',metrics=['verification/frozen_sft_reference','verification/reference_tensors'])]),
 ws.Section(name='Tool use and trajectories',panels=[
  wr.LinePlot(title='Tool errors and calls',x='episode',y=['episodes/tool_errors','episodes/tool_calls']),
  wr.LinePlot(title='Generated tokens',x='episode',y=['episodes/generated_tokens']),
  wr.MediaBrowser(title='Episode inspection',media_keys=['examples']),
  wr.MediaBrowser(title='Final comparison',media_keys=['comparison'])]),
 ws.Section(name='Resources and budget',panels=[
  wr.LinePlot(title='Estimated cost (not billing)',x='resources/elapsed_seconds',y=['budget/estimated_run_usd']),
  wr.LinePlot(title='Peak GPU memory (bytes)',x='optimizer_step',y=['resources/peak_gpu_bytes']),
  wr.LinePlot(title='Episode latency',x='episode',y=['episodes/seconds']),
  wr.LinePlot(title='Generation throughput',x='episode',y=['episodes/tokens_per_second'])])]
if state.exists() and read_json(state).get('entity')==entity:
 workspace=ws.Workspace.from_url(read_json(state)['url'])
 workspace.sections=sections
else:
 workspace=ws.Workspace(entity=entity,project=config['project'],name='Protein detective experiment',sections=sections)
workspace.save()
write_json(state,{'url':workspace.url,'entity':entity,'project':config['project']})
print(workspace.url)
