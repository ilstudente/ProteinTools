"""Shared tool protocol for scripted demonstrations, Transformers and GRPO."""
from __future__ import annotations
import json
import time
from .core import SYSTEM, parse_action, score, tool_text, qualifying


def messages_for(question):
    # Only the public question enters the policy context. No metadata or gold labels.
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question["text"]}]


def scripted_episode(snapshot, question):
    messages = messages_for(question)
    calls = 0
    def call(name, arguments):
        nonlocal calls
        calls += 1
        messages.append({"role": "assistant", "content": tool_text(name, arguments)})
        result = snapshot.dispatch(name, arguments)
        messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
        return result
    if question["query"] in snapshot.proteins:
        accession = question["query"]
    else:
        results = call("search_proteins", {"query": question["query"], "organism": question["organism"]})
        if len(results["results"]) != 1:
            raise ValueError("Ambiguous demonstration question")
        accession = results["results"][0]["accession"]
    candidates = call("search_structures", {"accession": accession, "method": question["method"],
                                           "resolution_below": question["threshold"]})
    if candidates["results"]:
        structure = call("get_structure", {"pdb_id": candidates["results"][0]["pdb_id"]})
        answer = {"status": "found", "uniprot_accession": accession, "pdb_id": structure["pdb_id"],
                  "resolution_angstrom": structure["resolution"]}
    else:
        answer = {"status": "no_match", "uniprot_accession": accession, "pdb_id": None, "resolution_angstrom": None}
    messages.append({"role": "assistant", "content": json.dumps(answer)})
    reward, failure = score(snapshot, question, answer)
    return {"question_id": question["id"], "question": question["text"], "split": question["split"],
            "accession": question["accession"], "kind": question["kind"], "messages": messages,
            "answer": answer, "reward": reward, "failure": failure, "tool_calls": calls, "tool_errors": 0}


class ModelRunner:
    def __init__(self, snapshot, tokenizer, config):
        self.snapshot, self.tokenizer, self.config = snapshot, tokenizer, config
        self.schemas = snapshot.schemas()
        # Evaluator-only cache: never attached to Snapshot tools or public messages.
        self._answer_sets = {}

    def prepare_answers(self, questions):
        for question in questions:
            key = (question["accession"], question["method"], question["threshold"])
            self._answer_sets[key] = qualifying(self.snapshot, question)

    def render(self, messages, generation=True):
        return self.tokenizer.apply_chat_template(messages, tools=self.schemas, tokenize=True,
                   add_generation_prompt=generation, enable_thinking=False, return_dict=False)

    def run(self, model, question, sample=False):
        import torch
        started = time.monotonic()
        key = (question["accession"], question["method"], question["threshold"])
        if key not in self._answer_sets:
            self.prepare_answers([question])
        messages = messages_for(question)
        initial = self.render(messages)
        ids, mask = list(initial), []
        calls = errors = generated = 0
        answer, failure = None, None
        was_training = model.training
        model.eval()
        try:
            for _ in range(self.config["max_tool_calls"] + 1):
                remaining = self.config["max_context"] - len(ids)
                if remaining <= 0:
                    failure = "context_limit"
                    break
                inputs = torch.tensor([ids], device=model.device)
                generation = {"max_new_tokens": min(self.config["max_turn_tokens"], remaining),
                              "do_sample": sample, "pad_token_id": self.tokenizer.pad_token_id,
                              "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True}
                if sample:
                    generation.update(temperature=self.config["temperature"], top_p=self.config["top_p"])
                with torch.no_grad():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **generation)
                new = output[0, len(ids):].tolist()
                ids.extend(new)
                mask.extend([1] * len(new))
                generated += len(new)
                text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
                messages.append({"role": "assistant", "content": text})
                if not new or new[-1] != self.tokenizer.eos_token_id:
                    failure = "generation_limit"
                    break
                try:
                    action, payload = parse_action(text)
                except (ValueError, TypeError):
                    failure = "invalid_action"
                    if "<tool_call>" in text:
                        calls += 1
                        errors += 1
                    break
                if action == "final":
                    answer = payload
                    break
                if calls >= self.config["max_tool_calls"]:
                    failure = "tool_limit"
                    break
                calls += 1
                result = self.snapshot.dispatch(payload["name"], payload["arguments"])
                errors += "error" in result
                messages.append({"role": "tool", "name": payload["name"], "content": json.dumps(result)})
                # Preserve sampled tokens exactly; append only environment framing and new generation prefix.
                # Qwen's tool message framing is independent of previous assistant content.
                suffix = '\n<|im_start|>user\n<tool_response>\n' + json.dumps(result) + '\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                if len(ids) + len(suffix_ids) >= self.config["max_context"]:
                    failure = "context_limit"
                    break
                ids.extend(suffix_ids)
                mask.extend([0] * len(suffix_ids))
            else:
                failure = "tool_limit"
        finally:
            model.train(was_training)
        reward, failure = score(self.snapshot, question, answer, failure, answer_set=self._answer_sets[key])
        return {"question_id": question["id"], "question": question["text"], "split": question["split"],
                "accession": question["accession"], "kind": question["kind"], "messages": messages,
                "answer": answer, "reward": reward, "failure": failure, "tool_calls": calls,
                "tool_errors": errors, "generated_tokens": generated, "seconds": time.monotonic() - started,
                "generation": {"sample": sample, "temperature": self.config["temperature"] if sample else 0,
                               "top_p": self.config["top_p"] if sample else None},
                "prompt_ids": initial, "completion_ids": ids[len(initial):], "env_mask": mask}


def training_example(tokenizer, snapshot, record, max_length=4096):
    """Expand each assistant turn into a causal example with all preceding context masked."""
    examples = []
    messages = record["messages"]
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(messages[:i], tools=snapshot.schemas(), tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False, return_dict=False)
        completion = tokenizer.encode(message["content"] + tokenizer.eos_token, add_special_tokens=False)
        ids = prompt + completion
        if len(ids) > max_length:
            raise ValueError("Demonstration exceeds context budget; never truncate silently")
        examples.append({"input_ids": ids, "attention_mask": [1]*len(ids), "labels": [-100]*len(prompt)+completion})
    return examples
