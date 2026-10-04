import json
import torch
from types import SimpleNamespace
from protein_detective.episodes import ModelRunner,training_example,scripted_episode
from protein_detective.core import tool_text

class Tokenizer:
    eos_token='§'
    eos_token_id=167
    pad_token_id=167
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(x) for x in ids if not(skip_special_tokens and x==167))
    def apply_chat_template(self,messages,**kwargs):
        text=''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:'
        return self.encode(text)

class Model:
    training=False
    device='cpu'
    def __init__(self,outputs):self.outputs=iter(outputs)
    def eval(self):self.training=False
    def train(self,v=True):self.training=v
    def generate(self,input_ids,**kwargs):
        output=next(self.outputs)+'§'
        return torch.cat([input_ids,torch.tensor([[ord(c) for c in output]])],dim=1)


def cfg(**overrides):
    return dict(max_context=4096,max_tool_calls=6,max_turn_tokens=384,temperature=.8,top_p=.95,**overrides)


def test_tool_outputs_masked_and_traces_retained(snapshot,question):
    final=json.dumps(dict(status='found',uniprot_accession='P00000',pdb_id='0000',resolution_angstrom=1.5))
    model=Model([tool_text('get_structure',{'pdb_id':'0000'}),final])
    r=ModelRunner(snapshot,Tokenizer(),cfg()).run(model,question)
    assert r['reward']==1 and r['tool_calls']==1
    assert 0 in r['env_mask'] and 1 in r['env_mask']
    assert len(r['env_mask'])==len(r['completion_ids'])
    assert sum(r['env_mask'])==r['generated_tokens']
    assert any(m['role']=='tool' for m in r['messages'])
    assert not model.training


def test_call_limit(snapshot,question):
    model=Model([tool_text('get_structure',{'pdb_id':'0000'})]*7)
    r=ModelRunner(snapshot,Tokenizer(),cfg()).run(model,question)
    assert r['reward']==0 and r['failure']=='tool_limit' and r['tool_calls']==6


def test_context_limit(snapshot,question):
    config=cfg();config['max_context']=1
    r=ModelRunner(snapshot,Tokenizer(),config).run(Model([]),question)
    assert r['failure']=='context_limit' and r['reward']==0


def test_sft_masks_all_prompt_context(snapshot,question):
    t=Tokenizer();r=scripted_episode(snapshot,question)
    examples=training_example(t,snapshot,r,max_length=10000)
    assert len(examples)==4
    for e,m in zip(examples,[m for m in r['messages'] if m['role']=='assistant']):
        supervised=[x for x in e['labels'] if x!=-100]
        assert t.decode(supervised)==m['content']+t.eos_token
        assert e['labels'][0]==-100


def test_precomputed_gold_stays_out_of_model_context(snapshot,question):
    runner=ModelRunner(snapshot,Tokenizer(),cfg())
    runner.prepare_answers([question])
    final=json.dumps(dict(status='found',uniprot_accession='P00000',pdb_id='0000',resolution_angstrom=1.5))
    r=runner.run(Model([final]),question)
    assert r['reward']==1
    assert 'answer_sets' not in json.dumps(r['messages'])
    assert '0000' not in json.dumps(r['messages'][:2])
