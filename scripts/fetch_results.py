"""Fetch experiment summaries and traces without downloading model/optimizer weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import modal
from protein_detective.core import read_json
config=read_json('configs/default.json')
volume=modal.Volume.from_name('protein-structure-detective')
remote=f"/runs/{config['experiment']}"
count=0
for entry in volume.iterdir(remote,recursive=True):
    path=Path(entry.path)
    if 'wandb' in path.parts or 'trainer' in path.parts or 'hf' in path.parts:
        continue
    if path.suffix not in {'.json','.gz','.md','.xml'}:
        continue
    local=Path(str(path).lstrip('/'))
    local.parent.mkdir(parents=True,exist_ok=True)
    with local.open('wb') as f:
        for chunk in volume.read_file(entry.path):f.write(chunk)
    count+=1
print(f'Downloaded {count} result and trace files')
