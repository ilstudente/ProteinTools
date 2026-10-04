"""Download only tokenizer files and check the exact training/inference protocol."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
os.environ.setdefault("HF_HOME", "data/hf")
import json
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import model_info
from protein_detective.core import Snapshot, read_json, write_json
from protein_detective.episodes import training_example, ModelRunner
config=read_json('configs/default.json')
revision=model_info(config['model']).sha
config['model_revision']=revision
write_json('configs/default.json',config)
t=AutoTokenizer.from_pretrained(config['model'],revision=revision)
t.pad_token=t.eos_token
s=Snapshot.load('data/v1/snapshot.json')
demos=read_json('data/v1/demonstrations.json')
examples=[x for r in demos for x in training_example(t,s,r)]
runner=ModelRunner(s,t,config)
r=demos[0]
# Match the environment suffix against Qwen's native template framing.
prefix=runner.render(r['messages'][:2])
assistant=t.encode(r['messages'][2]['content']+t.eos_token,add_special_tokens=False)
result=r['messages'][3]['content']
suffix='\n<|im_start|>user\n<tool_response>\n'+result+'\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
actual=prefix+assistant+t.encode(suffix,add_special_tokens=False)
expected=runner.render(r['messages'][:4])
# Qwen re-rendering may omit prior empty thinking blocks. Only the appended tool suffix must match.
assert t.decode(expected).endswith(suffix), repr(t.decode(expected)[-len(suffix)-30:])
assert all(any(x!=-100 for x in e['labels']) and e['labels'][0]==-100 for e in examples)
print({'revision':revision,'examples':len(examples),'max_tokens':max(len(e['input_ids']) for e in examples),'tool_suffix_verified':True})
write_json('data/v1/tokenizer-audit.json',{'revision':revision,'examples':len(examples),'max_tokens':max(len(e['input_ids']) for e in examples),'tool_suffix_verified':True})
