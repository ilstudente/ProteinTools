from protein_detective.core import write_json, Snapshot
from protein_detective.data import split_accessions, generate_questions
from protein_detective.episodes import scripted_episode
from protein_detective.evaluation import summarize


def test_shared_structure_never_crosses_splits(snapshot):
    snapshot.structures["0000"]["entities"].append({"accessions":["P00001"],"taxa":[9606]})
    splits=split_accessions(snapshot)
    owners={a:k for k,v in splits.items() for a in v}
    assert owners["P00000"]==owners["P00001"]
    assert len(owners)==12 and all(splits.values())
    assert splits==split_accessions(snapshot)


def test_generated_questions_all_solved(snapshot,tmp_path):
    write_json(tmp_path/"snapshot.json",{"proteins":list(snapshot.proteins.values()),"structures":list(snapshot.structures.values())})
    questions=generate_questions(tmp_path)
    assert len(questions)==120
    assert sum(q["kind"]=="no_match" for q in questions)==24
    assert all(scripted_episode(snapshot,q)["reward"]==1 for q in questions)
    assert questions==generate_questions(tmp_path)


def test_cluster_intervals():
    rows=[{"accession":a,"reward":v,"kind":"standard","tool_calls":2,"tool_errors":0} for a,v in [("A",1),("B",0)] for _ in range(5)]
    summary=summarize(rows)
    assert summary["success"]==.5 and summary["proteins"]==2
    assert summary["ci_low"]==0 and summary["ci_high"]==1


def test_threshold_text_preserves_full_precision(snapshot,tmp_path):
    for s in snapshot.matching('P00000'):s['resolution']=2.19650999049
    write_json(tmp_path/'snapshot.json',{'proteins':list(snapshot.proteins.values()),'structures':list(snapshot.structures.values())})
    questions=generate_questions(tmp_path)
    negatives=[q for q in questions if q['accession']=='P00000' and q['kind']=='no_match']
    assert len(negatives)==2
    assert all('2.19650999049' in q['text'] for q in negatives)


def test_manifest_detects_mutation_and_heldout_demo(snapshot,tmp_path):
    import pytest
    from protein_detective.data import digest,verify_dataset
    from protein_detective.core import read_json
    raw={'proteins':list(snapshot.proteins.values()),'structures':list(snapshot.structures.values())}
    write_json(tmp_path/'snapshot.json',raw)
    qs=generate_questions(tmp_path)
    demos=[scripted_episode(snapshot,next(q for q in qs if q['split']=='train'))]
    write_json(tmp_path/'demonstrations.json',demos)
    manifest={'snapshot_hash':digest(raw),'questions_hash':digest(qs),
              'splits_hash':digest(read_json(tmp_path/'splits.json')),'demonstrations_hash':digest(demos)}
    write_json(tmp_path/'manifest.json',manifest)
    verify_dataset(tmp_path)
    raw['structures'][0]['resolution']=9
    write_json(tmp_path/'snapshot.json',raw)
    with pytest.raises(ValueError,match='Snapshot hash'):verify_dataset(tmp_path)
    manifest['snapshot_hash']=digest(raw)
    heldout=scripted_episode(snapshot,next(q for q in qs if q['split']=='test'))
    write_json(tmp_path/'demonstrations.json',[heldout])
    manifest['demonstrations_hash']=digest([heldout])
    write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Held-out demonstration'):verify_dataset(tmp_path)


def test_paired_comparison_requires_identical_questions():
    import pytest
    from protein_detective.evaluation import paired_gain
    before=[dict(question_id=str(i),question=f'q{i}',accession=str(i),reward=0) for i in range(2)]
    after=[dict(r,reward=int(i==0)) for i,r in enumerate(before)]
    result=paired_gain(before,after)
    assert result==dict(gain=.5,ci_low=0,ci_high=1)
    after[0]['question']='changed question'
    with pytest.raises(ValueError,match='changed'):paired_gain(before,after)
