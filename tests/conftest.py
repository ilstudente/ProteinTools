import pytest
from protein_detective.core import Snapshot, METHOD

@pytest.fixture
def snapshot():
    proteins=[{"accession":f"P{i:05d}","name":f"Protein {i}","aliases":[f"GENE{i}"],"taxon":9606,
               "organism":"Homo sapiens","organism_aliases":["human"],"sequence":"MAA"} for i in range(12)]
    structures=[]
    for i,p in enumerate(proteins):
        for j in range(7):
            structures.append({"pdb_id":f"{i:02d}{j:02d}","methods":[METHOD],"resolution":1.5+j*.2,
                "entities":[{"entity_id":f"{i}_{j}","accessions":[p["accession"]],"taxa":[9606]}]})
    return Snapshot({"proteins":proteins,"structures":structures})

@pytest.fixture
def question():
    return {"id":"q1","text":"Find Protein 0 from human below 2 angstroms in the frozen collection.",
            "accession":"P00000","query":"Protein 0","organism":"human","method":METHOD,
            "threshold":2.0,"kind":"standard","split":"train"}
