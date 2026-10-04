import json
import math
import pytest
from protein_detective.core import score, qualifying, parse_action, tool_text
from protein_detective.episodes import scripted_episode,messages_for


def answer(pdb="0000",resolution=1.5):
    return dict(status="found",uniprot_accession="P00000",pdb_id=pdb,resolution_angstrom=resolution)


def test_threshold_and_multiple_answers(snapshot,question):
    assert score(snapshot,question,answer())[0]==1
    assert score(snapshot,question,answer("0001",1.7))[0]==1
    question["threshold"]=1.5
    assert score(snapshot,question,answer())[0]==0
    assert score(snapshot,question,dict(status="no_match",uniprot_accession="P00000",pdb_id=None,resolution_angstrom=None))[0]==1


@pytest.mark.parametrize("resolution",[None,True,float("nan"),float("inf"),1.8,"1.5"])
def test_bad_resolution(snapshot,question,resolution):
    assert score(snapshot,question,answer(resolution=resolution))[0]==0


def test_missing_resolution(snapshot,question):
    snapshot.structures["0000"]["resolution"]=None
    assert "0000" not in qualifying(snapshot,question)


def test_entity_organism_mismatch(snapshot,question):
    s=snapshot.structures["0000"]
    s["entities"][0]["taxa"]=[10090]
    s["entities"].append({"accessions":["P00001"],"taxa":[9606]})
    assert score(snapshot,question,answer())[0]==0


def test_ambiguous_entity_excluded(snapshot,question):
    snapshot.structures["0000"]["entities"][0]["accessions"].append("P00001")
    assert score(snapshot,question,answer())[0]==0


def test_schema_and_protein(snapshot,question):
    a=answer();a["extra"]=True
    assert score(snapshot,question,a)[1]=="invalid_schema"
    a=answer();a["uniprot_accession"]="P00001"
    assert score(snapshot,question,a)[1]=="wrong_protein"
    assert score(snapshot,question,"not json")[1]=="invalid_json"


def test_pagination_and_empty(snapshot):
    first=snapshot.search_structures("P00000")
    assert len(first["results"])==5 and first["next_page"]==2 and first["total"]==7
    second=snapshot.search_structures("P00000",page=2)
    assert len(second["results"])==2 and second["next_page"] is None
    assert snapshot.search_structures("P00000",resolution_below=1)["total"]==0
    assert snapshot.search_structures("P00000",page=0)["error"]=="invalid_page"


def test_tool_errors(snapshot):
    assert snapshot.dispatch("bogus",{})["error"]=="unknown_tool"
    assert snapshot.dispatch("get_structure",{})["error"]=="invalid_arguments"
    assert snapshot.dispatch("get_structure",{"pdb_id":[]})["error"]=="invalid_arguments"
    assert snapshot.dispatch("get_structure",{"pdb_id":"bad"})["error"]=="not_found"
    assert snapshot.search_proteins("gene0","HUMAN")["results"][0]["accession"]=="P00000"


def test_protocol():
    assert parse_action(tool_text("get_structure",{"pdb_id":"0000"}))[0]=="tool"
    with pytest.raises(ValueError):parse_action('<tool_call>{"name":"a"}</tool_call>')
    with pytest.raises(ValueError):parse_action(tool_text("a",{})+tool_text("b",{}))


def test_solver_and_gold_isolation(snapshot,question):
    record=scripted_episode(snapshot,question)
    assert record["reward"]==1 and record["tool_calls"]==3
    public=messages_for(question)
    assert question["accession"] not in json.dumps(public)
    assert "threshold" not in json.dumps(public)
    assert score(snapshot,question,answer(),"tool_limit")[0]==0


def test_tool_matching_enforces_same_organism_as_reward(snapshot,question):
    snapshot.structures['0000']['entities'][0]['taxa']=[10090]
    assert {'pdb_id':'0000'} not in snapshot.search_structures('P00000')['results']
    assert '0000' not in qualifying(snapshot,question)
