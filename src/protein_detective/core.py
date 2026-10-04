"""Pure, dependency-light snapshot tools and strict answer verification."""
from __future__ import annotations
import inspect
import json
import math
import re
from pathlib import Path

METHOD = "X-RAY DIFFRACTION"
SYSTEM = '''You are a protein structure detective. Answer using only the frozen collection.
Use the supplied tools to resolve the protein and check experimental structures.
A resolution better than X means strictly less than X angstroms. Any qualifying structure is acceptable.
Make one tool call at a time using <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>.
Return only a JSON object as your final answer with exactly these fields:
status (found or no_match), uniprot_accession, pdb_id, resolution_angstrom.
For no_match, pdb_id and resolution_angstrom must be null. Do not invent identifiers.
You may make at most six tool calls. Do not output reasoning or think tags.'''


def norm(s):
    return re.sub(r"\s+", " ", str(s).casefold()).strip()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


class Snapshot:
    def __init__(self, data):
        self.proteins = {p["accession"]: p for p in data["proteins"]}
        self.structures = {s["pdb_id"]: s for s in data["structures"]}

    @classmethod
    def load(cls, path):
        return cls(read_json(path))

    def matching(self, accession):
        return [s for s in self.structures.values() if any(
            e["accessions"] == [accession] and self.proteins[accession]["taxon"] in e["taxa"] for e in s["entities"])]

    @staticmethod
    def page(items, page):
        if type(page) is not int or page < 1:
            return {"error": "invalid_page", "message": "page must be a positive integer"}
        start = (page - 1) * 5
        return {"results": items[start:start + 5], "total": len(items), "page": page,
                "next_page": page + 1 if start + 5 < len(items) else None}

    def search_proteins(self, query: str, organism: str, page: int = 1) -> dict:
        """Find proteins by accession, name or alias in the frozen collection.

        Args:
            query: Protein name, alias, gene name, or accession.
            organism: Common or scientific organism name, or taxonomy ID.
            page: One-based page number; each page contains at most five records.
        """
        if not isinstance(query, str) or not query.strip() or not isinstance(organism, str):
            return {"error": "invalid_arguments"}
        matches = []
        for p in self.proteins.values():
            if norm(organism) not in {norm(x) for x in [p["organism"], str(p["taxon"]), *p["organism_aliases"]]}:
                continue
            names = [p["accession"], p["name"], *p["aliases"]]
            if any(norm(query) in norm(x) for x in names):
                rank = 0 if any(norm(query) == norm(x) for x in names) else 1
                matches.append((rank, p["accession"], {k: p[k] for k in ["accession", "name", "aliases", "organism", "taxon"]}))
        return self.page([m[2] for m in sorted(matches)], page)

    def search_structures(self, accession: str, method: str | None = None,
                          resolution_below: float | None = None, page: int = 1) -> dict:
        """Find structure identifiers mapped to a protein in the frozen collection.

        Args:
            accession: Exact UniProt accession.
            method: Optional experimental method, e.g. X-RAY DIFFRACTION.
            resolution_below: Optional strict upper resolution bound in angstroms.
            page: One-based page number.
        """
        if accession not in self.proteins:
            return {"error": "not_found", "message": "Unknown protein accession"}
        if resolution_below is not None and (type(resolution_below) not in (int, float) or
                not math.isfinite(resolution_below) or resolution_below <= 0):
            return {"error": "invalid_resolution"}
        rows = []
        for s in self.matching(accession):
            if method is not None and norm(method) not in {norm(m) for m in s["methods"]}:
                continue
            if resolution_below is not None and (s["resolution"] is None or s["resolution"] >= resolution_below):
                continue
            rows.append({"pdb_id": s["pdb_id"]})
        return self.page(sorted(rows, key=lambda r: r["pdb_id"]), page)

    def get_structure(self, pdb_id: str) -> dict:
        """Inspect experimental metadata and protein mappings for one structure.

        Args:
            pdb_id: PDB identifier returned by search_structures.
        """
        return self.structures.get(pdb_id.upper(), {"error": "not_found"})

    def dispatch(self, name, arguments):
        if name not in {"search_proteins", "search_structures", "get_structure"}:
            return {"error": "unknown_tool"}
        if not isinstance(arguments, dict):
            return {"error": "invalid_arguments"}
        try:
            fn = getattr(self, name)
            inspect.signature(fn).bind(**arguments)
            return fn(**arguments)
        except (TypeError, ValueError, AttributeError, KeyError):
            return {"error": "invalid_arguments"}

    def schemas(self):
        def schema(name, description, properties, required):
            return {"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required,
                               "additionalProperties": False}}}
        string = {"type": "string"}
        page = {"type": "integer", "minimum": 1, "default": 1}
        return [
            schema("search_proteins", "Find protein names, aliases or accessions for an organism.",
                   {"query": string, "organism": string, "page": page}, ["query", "organism"]),
            schema("search_structures", "Find PDB IDs mapped to a UniProt accession; resolution_below is strict.",
                   {"accession": string, "method": string, "resolution_below": {"type": "number"}, "page": page}, ["accession"]),
            schema("get_structure", "Inspect method, resolution and protein entity mappings.", {"pdb_id": string}, ["pdb_id"])]


def qualifying(snapshot, question):
    p = snapshot.proteins[question["accession"]]
    return {s["pdb_id"]: s["resolution"] for s in snapshot.matching(p["accession"])
            if question["method"] in s["methods"] and s["resolution"] is not None
            and s["resolution"] < question["threshold"] and any(
                e["accessions"] == [p["accession"]] and p["taxon"] in e["taxa"] for e in s["entities"])}


def score(snapshot, question, answer, failure=None, *, answer_set=None):
    if failure:
        return 0.0, failure
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except (ValueError, TypeError):
            return 0.0, "invalid_json"
    if not isinstance(answer, dict) or set(answer) != {"status", "uniprot_accession", "pdb_id", "resolution_angstrom"}:
        return 0.0, "invalid_schema"
    if answer["uniprot_accession"] != question["accession"]:
        return 0.0, "wrong_protein"
    valid = qualifying(snapshot, question) if answer_set is None else answer_set
    if answer["status"] == "no_match":
        ok = not valid and answer["pdb_id"] is None and answer["resolution_angstrom"] is None
        return (1.0, None) if ok else (0.0, "incorrect_no_match")
    if answer["status"] != "found":
        return 0.0, "invalid_status"
    if not isinstance(answer["pdb_id"], str) or answer["pdb_id"] not in valid:
        return 0.0, "nonqualifying_structure"
    r = answer["resolution_angstrom"]
    if type(r) not in (int, float) or not math.isfinite(r) or abs(r - valid[answer["pdb_id"]]) > 0.001:
        return 0.0, "wrong_resolution"
    return 1.0, None


def parse_action(text):
    text = text.strip()
    if text.startswith("<tool_call>") and text.endswith("</tool_call>") and text.count("<tool_call>") == 1:
        call = json.loads(text[len("<tool_call>"):-len("</tool_call>")])
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("invalid tool call")
        return "tool", call
    return "final", json.loads(text)


def tool_text(name, arguments):
    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'
