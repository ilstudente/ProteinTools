"""Resumable public metadata collection, connected-component splits and questions."""
from __future__ import annotations
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from .core import METHOD, Snapshot, norm, qualifying, write_json, read_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class CachedHTTP:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ProteinStructureDetective/0.1 (research snapshot collector)"
        self.failures = 0

    def get(self, url, params=None, body=None):
        key = digest({"url": url, "params": params, "body": body})
        path = self.directory / f"{key}.json"
        if path.exists():
            return read_json(path)
        for attempt in range(5):
            try:
                time.sleep(0.12)
                response = self.session.request("POST" if body is not None else "GET", url,
                                                params=params, json=body, timeout=90)
                response.raise_for_status()
                payload = response.json() if response.content else {}
                if isinstance(payload, dict) and payload.get("errors"):
                    raise ValueError(str(payload["errors"]))
                result = {"url": response.url, "body": body, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                          "links": response.links, "data": payload}
                write_json(path, result)
                return result
            except (requests.RequestException, ValueError):
                self.failures += 1
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))


def collect(directory, count=100, seed=42):
    directory = Path(directory)
    http = CachedHTTP(directory / "raw")
    proteins = []
    for taxon, common, scientific, quota in [(9606, "human", "Homo sapiens", (count + 2)//3),
            (10090, "mouse", "Mus musculus", (count + 1)//3), (562, "E. coli", "Escherichia coli", count//3)]:
        # E. coli strains are descendants of taxon 562; retain each actual taxon in normalized records.
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": f"reviewed:true AND taxonomy_id:{taxon} AND database:pdb", "format": "json", "size": 100}
        candidates = []
        while url and len(candidates) < quota * 3:
            page = http.get(url, params=params)
            candidates.extend(page["data"]["results"])
            url = page["links"].get("next", {}).get("url")
            params = None
        rng = random.Random(seed + taxon)
        rng.shuffle(candidates)
        accepted = 0
        for p in candidates:
            desc = p.get("proteinDescription", {})
            name = desc.get("recommendedName", {}).get("fullName", {}).get("value")
            if not name:
                continue
            aliases = []
            for section in [desc.get("recommendedName", {}), *desc.get("alternativeNames", [])]:
                aliases.extend(x["value"] for x in section.get("shortNames", []))
                if section.get("fullName", {}).get("value") != name and section.get("fullName"):
                    aliases.append(section["fullName"]["value"])
            for gene in p.get("genes", []):
                aliases.extend(x["value"] for x in [gene.get("geneName", {}), *gene.get("synonyms", [])] if "value" in x)
            protein = {"accession": p["primaryAccession"], "name": name, "aliases": sorted(set(aliases)),
                       "taxon": p["organism"]["taxonId"], "organism": p["organism"]["scientificName"],
                       "organism_aliases": [common, scientific], "sequence": p["sequence"]["value"]}
            ids = structure_ids(http, protein["accession"])
            if not ids:
                continue
            protein["source_pdb_ids"] = ids
            proteins.append(protein)
            accepted += 1
            print(f"Collected protein {len(proteins)}/{count}: {protein['accession']} ({len(ids)} structures)", flush=True)
            if accepted == quota:
                break
        if accepted < quota:
            raise RuntimeError(f"Not enough experimental proteins for {scientific}; expand candidate collection")
    all_ids = sorted({x for p in proteins for x in p["source_pdb_ids"]})
    structures = []
    exclusions = {"ambiguous_entities": 0, "missing_resolution": 0}
    for start in range(0, len(all_ids), 50):
        ids = all_ids[start:start + 50]
        query = '''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id exptl{method}
          rcsb_entry_info{resolution_combined structure_determination_methodology}
          polymer_entities{rcsb_id rcsb_polymer_entity_container_identifiers{
          reference_sequence_identifiers{database_name database_accession}}
          rcsb_entity_source_organism{ncbi_taxonomy_id}}}}'''
        data = http.get("https://data.rcsb.org/graphql", body={"query": query, "variables": {"ids": ids}})["data"]
        for entry in data["data"]["entries"]:
            if not entry:
                continue
            methods = [x["method"] for x in entry.get("exptl") or []]
            info = entry.get("rcsb_entry_info") or {}
            if METHOD not in methods or info.get("structure_determination_methodology") != "experimental":
                continue
            rs = info.get("resolution_combined") or []
            resolution = min(rs) if rs else None
            exclusions["missing_resolution"] += resolution is None
            entities = []
            for e in entry.get("polymer_entities") or []:
                refs = (e.get("rcsb_polymer_entity_container_identifiers") or {}).get("reference_sequence_identifiers") or []
                accs = sorted({r["database_accession"] for r in refs if r["database_name"] == "UniProt"})
                exclusions["ambiguous_entities"] += len(accs) > 1
                entities.append({"entity_id": e["rcsb_id"], "accessions": accs,
                                 "taxa": sorted({x["ncbi_taxonomy_id"] for x in e.get("rcsb_entity_source_organism") or [] if x.get("ncbi_taxonomy_id") is not None})})
            structures.append({"pdb_id": entry["rcsb_id"], "methods": methods, "resolution": resolution, "entities": entities})
        print(f"Downloaded metadata {min(start+50,len(all_ids))}/{len(all_ids)}", flush=True)
    snapshot = {"proteins": proteins, "structures": sorted(structures, key=lambda s: s["pdb_id"])}
    write_json(directory / "snapshot.json", snapshot)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "snapshot_hash": digest(snapshot),
                "proteins": len(proteins), "structures": len(structures), "exclusions": exclusions,
                "collection_failures": http.failures, "seed": seed,
                "raw_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((directory / "raw").glob("*.json"))}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def structure_ids(http, accession):
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", "operator": "exact_match", "value": accession}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": METHOD}}]
    result, start = [], 0
    while True:
        query = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": "entry",
                 "request_options": {"results_content_type": ["experimental"], "paginate": {"start": start, "rows": 1000},
                                     "sort": [{"sort_by": "rcsb_id", "direction": "asc"}]}}
        page = http.get("https://search.rcsb.org/rcsbsearch/v2/query", body=query)["data"]
        rows = page.get("result_set", [])
        result.extend(x["identifier"] for x in rows)
        if len(result) >= page.get("total_count", 0):
            return sorted(set(result))
        if not rows:
            raise RuntimeError("Unexpected empty PDB page before total_count")
        start += len(rows)


def split_accessions(snapshot, seed=42):
    # Connected components prevent a multi-protein structure crossing splits.
    parent = {a: a for a in snapshot.proteins}
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s in snapshot.structures.values():
        accs = [a for e in s["entities"] for a in e["accessions"] if a in parent]
        for a in accs[1:]:
            parent[root(a)] = root(accs[0])
    groups = {}
    for a in parent:
        groups.setdefault(root(a), []).append(a)
    blocks = list(groups.values())
    random.Random(seed).shuffle(blocks)
    blocks.sort(key=len, reverse=True)
    result = {"train": [], "validation": [], "test": []}
    targets = {"train": .7 * len(parent), "validation": .15 * len(parent), "test": .15 * len(parent)}
    for block in blocks:
        dest = max(result, key=lambda k: targets[k] - len(result[k]))
        result[dest].extend(block)
    if any(not v for v in result.values()):
        raise ValueError("Too few independent protein components for three splits")
    return {k: sorted(v) for k, v in result.items()}


def generate_questions(directory, seed=42):
    directory = Path(directory)
    snap = Snapshot.load(directory / "snapshot.json")
    splits = split_accessions(snap, seed)
    write_json(directory / "splits.json", splits)
    questions = []
    templates = ["In the frozen collection, find an experimental X-ray structure of {name} from {organism} with resolution better than {threshold} Å.",
                 "Using only the frozen collection, return an X-ray structure for {organism} {name} at a resolution strictly below {threshold} angstroms.",
                 "Find any frozen-collection X-ray structure for {name} ({organism}) with resolution < {threshold} Å, or report no_match."]
    for split, accs in splits.items():
        for a in accs:
            p = snap.proteins[a]
            rs = [s["resolution"] for s in snap.matching(a) if s["resolution"] is not None and
                  any(e["accessions"] == [a] and p["taxon"] in e["taxa"] for e in s["entities"])]
            if not rs:
                continue
            aliases = [x for x in p["aliases"] if len(snap.search_proteins(x, p["organism"])["results"]) == 1]
            # Alternate per-protein 3/2 and 2/1 alias/accession examples for global 25%/15%.
            odd = len(questions)//10 % 2
            types = ["standard"]*4 + ["alias"]*(2+odd) + ["accession"]*(2-odd) + ["no_match"]*2
            for i, kind in enumerate(types):
                if kind == "alias" and not aliases:
                    kind = "standard"
                name = a if kind == "accession" else (aliases[i % len(aliases)] if kind == "alias" else p["name"])
                # Avoid ambiguous natural-language identities.
                matches = snap.search_proteins(name, p["organism"])["results"]
                if kind != "accession" and (len(matches) != 1 or matches[0]["accession"] != a):
                    name = a
                    if kind != "no_match":
                        kind = "accession"
                threshold = min(rs) if kind == "no_match" else round(min(rs) + [0.1, 0.3, 0.5, 1.0][i % 4], 4)
                q = {"id": f"{a}-{i:02d}", "split": split, "kind": kind, "accession": a,
                     "query": name, "organism": p["organism"], "method": METHOD, "threshold": threshold,
                     "text": templates[i % len(templates)].format(name=name, organism=p["organism"], threshold=format(threshold,"g") if float(format(threshold,"g"))==threshold else str(threshold))}
                assert bool(qualifying(snap, q)) == (kind != "no_match")
                questions.append(q)
    write_json(directory / "questions.json", questions)
    return questions


def verify_dataset(directory):
    directory=Path(directory)
    manifest=read_json(directory/'manifest.json')
    snapshot=read_json(directory/'snapshot.json')
    if digest(snapshot)!=manifest['snapshot_hash']:
        raise ValueError('Snapshot hash mismatch')
    for filename,key in [('questions.json','questions_hash'),('splits.json','splits_hash'),('demonstrations.json','demonstrations_hash')]:
        if key not in manifest or digest(read_json(directory/filename))!=manifest[key]:
            raise ValueError(f'{filename} hash mismatch or missing manifest hash')
    splits=read_json(directory/'splits.json')
    owner={}
    for split,accessions in splits.items():
        for a in accessions:
            if a in owner:raise ValueError('Protein occurs in multiple splits')
            owner[a]=split
    for s in snapshot['structures']:
        if len({owner[a] for e in s['entities'] for a in e['accessions'] if a in owner})>1:
            raise ValueError('PDB structure crosses splits')
    for q in read_json(directory/'questions.json'):
        if owner[q['accession']]!=q['split']:raise ValueError('Question split mismatch')
    for r in read_json(directory/'demonstrations.json'):
        if owner[r['accession']]!='train':raise ValueError('Held-out demonstration leakage')
    return manifest
