"""Pinned TRL integration. No API calls to biology services during training."""
from __future__ import annotations
import json
import time
from pathlib import Path
from .core import Snapshot, read_json, write_json
from .episodes import ModelRunner, training_example
from .tracking import Tracker


def load_model(config, checkpoint=None, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from huggingface_hub import model_info
    from peft import LoraConfig, PeftModel, get_peft_model
    set_seed(config["seed"])
    revision = config.get("model_revision") or model_info(config["model"]).sha
    config["model_revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint:
        metadata = read_json(Path(checkpoint)/"lineage.json")
        if metadata["model"] != config["model"] or metadata["model_revision"] != revision:
            raise ValueError("Checkpoint base model revision mismatch")
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                lora_dropout=0, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    return model, tokenizer


def lineage(data, config, checkpoint=None):
    data = Path(data)
    info = {"model":config["model"], "model_revision":config["model_revision"],
            "manifest":read_json(data/"manifest.json")}
    if (data/"prepared-artifact.json").exists():
        info["dataset_artifact"] = read_json(data/"prepared-artifact.json")
    if checkpoint:
        parent=read_json(Path(checkpoint)/"lineage.json")
        for key in ["snapshot_hash","questions_hash","splits_hash"]:
            if parent["manifest"].get(key) != info["manifest"].get(key):
                raise ValueError(f"Checkpoint dataset mismatch: {key}")
        info["parent_checkpoint"] = read_json(Path(checkpoint)/"artifact.json") if (Path(checkpoint)/"artifact.json").exists() else str(checkpoint)
    return info


def save_adapter(model, tokenizer, path, tracker, info, name):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, selected_adapters=["default"])
    tokenizer.save_pretrained(path)
    write_json(path/"lineage.json", info)
    tracker.flush()
    ref = tracker.artifact(path, name, "model")
    write_json(path/"artifact.json", ref)
    return path


def install_sft_reference(model, tracker=None):
    """Make TRL's frozen "ref" adapter a bit-exact copy of the trained adapter.

    Pinned TRL 0.28 creates the reference adapter with add_adapter and copies the
    trained weights in place. PEFT keeps a trainable adapter in fp32 on a bf16 base,
    while the new adapter is allocated in bf16, so that copy rounds every weight.
    Replace the reference tensors with exact clones so the KL term measures drift
    from the SFT policy rather than from a rounded approximation of it.
    """
    import torch
    assert "ref" in model.peft_config, "KL must reference SFT, never the untouched base"
    parameters = dict(model.named_parameters())
    count = 0
    for name, parameter in parameters.items():
        if "lora_" in name and ".default." in name:
            reference = parameters[name.replace(".default.", ".ref.")]
            reference.requires_grad_(False)
            reference.data = parameter.data.detach().clone()
            count += 1
    assert count, "No trainable LoRA tensors found"
    for name, parameter in parameters.items():
        if "lora_" in name and ".default." in name:
            reference = parameters[name.replace(".default.", ".ref.")]
            assert reference.dtype == parameter.dtype, "Reference adapter dtype differs"
            assert torch.equal(parameter.detach(), reference.detach()), "SFT reference weights differ"
            assert not reference.requires_grad, "Reference adapter must be frozen"
            assert parameter.requires_grad, "Trained adapter must remain trainable"
    if tracker is not None:
        tracker.log({"verification/frozen_sft_reference": 1, "verification/reference_tensors": count})
    return count


def make_grpo_class():
    import torch
    from trl import GRPOTrainer
    from trl.models.utils import unwrap_model_for_generation
    class DetectiveGRPOTrainer(GRPOTrainer):
        """Supply bounded Transformers multi-turn rollouts to pinned TRL 0.28.

        TRL's public rollout_func is routed through vLLM in this release. Override
        only the generation hook; retain native scoring, masking, KL and optimizer.
        env_mask is consumed by GRPOTrainer._generate and masks environment tokens.
        """
        def __init__(self, *args, runner, question_map, tracker, **kwargs):
            self.runner, self.question_map, self.tracker = runner, question_map, tracker
            super().__init__(*args, **kwargs)

        def _generate_single_turn(self, prompts):
            rows = []
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as model:
                for prompt in prompts:
                    q = self.question_map[prompt]
                    r = self.runner.run(model, q, sample=True)
                    r["checkpoint"] = f"optimizer-step-{self.state.global_step}"
                    self.tracker.episode(r)
                    rows.append(r)
            groups = [rows[i:i+self.num_generations] for i in range(0,len(rows),self.num_generations)]
            self.tracker.log({"optimizer_step": self.state.global_step,
                "learning/mixed_groups":sum(0<sum(r["reward"] for r in g)<len(g) for g in groups)/len(groups),
                "learning/all_zero_groups":sum(all(r["reward"]==0 for r in g) for g in groups)/len(groups),
                "learning/all_one_groups":sum(all(r["reward"]==1 for r in g) for g in groups)/len(groups)})
            return ([r["prompt_ids"] for r in rows], [r["completion_ids"] for r in rows], None,
                    {"env_mask":[r["env_mask"] for r in rows], "episode_reward":[r["reward"] for r in rows]})
    return DetectiveGRPOTrainer


def train(data, runs, config, stage, checkpoint=None, suffix=None):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTTrainer, SFTConfig, GRPOConfig
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from .evaluation import evaluate, readiness
    data, runs = Path(data), Path(runs)
    from .data import verify_dataset
    verify_dataset(data)
    snapshot = Snapshot.load(data/"snapshot.json")
    questions = read_json(data/"questions.json")
    train_q = [q for q in questions if q["split"]=="train"]
    validation = [q for q in questions if q["split"]=="validation"]
    if stage=="grpo" and not checkpoint:
        raise ValueError("GRPO requires the SFT adapter checkpoint")
    if stage=="baseline":
        return evaluate_checkpoint(data, runs, config, None, "validation", "original")
    completed_adapter = runs/config["experiment"]/(suffix or stage)/"final"
    if stage == "sft" and (completed_adapter/"artifact.json").exists():
        # Saving final occurs only after trainer.train completes. Recover validation
        # without performing another epoch or changing the checkpoint identity.
        model, tokenizer = load_model(config, completed_adapter, trainable=False)
        lineage(data, config, completed_adapter)
        with Tracker(runs, config, stage, suffix) as tracker:
            tracker.run.use_artifact(read_json(completed_adapter/"artifact.json")["name"])
            states = sorted((tracker.root/"trainer").glob("checkpoint-*/trainer_state.json"),
                            key=lambda p: int(p.parent.name.split("-")[-1]))
            step = read_json(states[-1])["global_step"] if states else 0
            runner = ModelRunner(snapshot, tokenizer, config)
            runner.prepare_answers(validation)
            summary, records = evaluate(model, runner, validation,
                                        tracker, str(completed_adapter), step)
            write_json(tracker.root/"validation-summary.json", summary)
            write_json(tracker.root/"validation-records.json", records)
            tracker.artifact(tracker.root/"validation-summary.json",f"validation-{tracker.run_id}","evaluation")
            tracker.artifact(tracker.root/"validation-records.json",f"validation-records-{tracker.run_id}","evaluation")
        return str(completed_adapter)
    model, tokenizer = load_model(config, checkpoint, trainable=True)
    runner = ModelRunner(snapshot, tokenizer, config)
    runner.prepare_answers(questions)
    info = lineage(data, config, checkpoint)
    with Tracker(runs, config, stage, suffix) as tracker:
        tracker.run.config.update({**info, "tool_schemas": snapshot.schemas(), "system_prompt": __import__('protein_detective.core',fromlist=['SYSTEM']).SYSTEM,
                                  "chat_template":tokenizer.chat_template, "enable_thinking":False}, allow_val_change=True)
        if "dataset_artifact" in info and not info["dataset_artifact"].get("pending_sync"):
            tracker.run.use_artifact(info["dataset_artifact"]["name"])
        if isinstance(info.get("parent_checkpoint"),dict):
            tracker.run.use_artifact(info["parent_checkpoint"]["name"])
        tracker.run.config.update({"device":str(model.device), "cuda":torch.version.cuda,
                                  "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}, allow_val_change=True)
        out = tracker.root/"trainer"
        started = time.monotonic()
        class Lifecycle(TrainerCallback):
            best = -1
            validation_seconds = 0
            def on_step_end(self, args, state, control, **kwargs):
                elapsed = time.monotonic()-started
                # Leave time for a final validation and checkpoint upload.
                reserve = max(180, self.validation_seconds * 1.5 + 120)
                limit = config.get("job_timeout_seconds", 3600)-reserve
                if elapsed >= limit:
                    control.should_training_stop = True
                    control.should_save = True
                if torch.cuda.is_available():
                    tracker.log({"optimizer_step":state.global_step,
                                 "resources/peak_gpu_bytes":torch.cuda.max_memory_allocated()})
                # The post-training evaluation covers the final step; avoid a duplicate 140-episode pass.
                if stage=="grpo" and state.global_step % config["eval_steps"]==0 and state.global_step < state.max_steps and not control.should_training_stop:
                    self.validate(state.global_step)
                return control
            def validate(self, step):
                validation_started = time.monotonic()
                summary, records = evaluate(model, runner, validation, tracker, f"step-{step}", step)
                self.validation_seconds = time.monotonic() - validation_started
                write_json(tracker.root/f"validation-{step}.json",summary)
                # Validation-only selection; ties go to the later checkpoint because validation cannot separate them.
                if summary["success"] >= self.best:
                    self.best = summary["success"]
                    save_adapter(model,tokenizer,tracker.root/"best",tracker,info,f"{config['experiment']}-{stage}-best")
            def on_save(self,args,state,control,**kwargs):
                tracker.flush()
                cp=Path(args.output_dir)/f"checkpoint-{state.global_step}"
                if cp.exists():
                    write_json(cp/"lineage.json",info)
                    tracker.artifact(cp,f"{config['experiment']}-{stage}-recovery","checkpoint")
        lifecycle=Lifecycle()
        previous=[read_json(p)["success"] for p in tracker.root.glob("validation-*.json") if p.name != "validation-summary.json"]
        lifecycle.best=max(previous,default=-1)
        common = dict(output_dir=str(out), per_device_train_batch_size=1, bf16=torch.cuda.is_available(),
                      gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
                      logging_steps=1, report_to="wandb", seed=config["seed"], save_total_limit=2,
                      optim="adamw_torch", dataloader_num_workers=0)
        if stage in {"sft","smoke"}:
            demos=read_json(data/"demonstrations.json")
            if stage=="smoke": demos=demos[:4]
            examples=[e for r in demos for e in training_example(tokenizer,snapshot,r,config["max_context"])]
            mask_audit={"examples":len(examples), "supervised_tokens":sum(sum(x!=-100 for x in e["labels"]) for e in examples),
                        "masked_tokens":sum(sum(x==-100 for x in e["labels"]) for e in examples)}
            write_json(tracker.root/"mask-audit.json",mask_audit)
            tracker.artifact(tracker.root/"mask-audit.json",f"mask-audit-{tracker.run_id}","verification")
            args=SFTConfig(**common,learning_rate=config["sft_learning_rate"],num_train_epochs=config["sft_epochs"],
                    gradient_accumulation_steps=2 if stage=="smoke" else config["sft_batch_size"],
                    max_steps=1 if stage=="smoke" else -1,max_length=config["max_context"],packing=False,
                    dataset_kwargs={"skip_prepare_dataset":True},save_strategy="epoch")
            trainer=SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list(examples),processing_class=tokenizer,
                data_collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),callbacks=[lifecycle])
        else:
            prior=read_json(Path(checkpoint).parent/"validation-summary.json")
            if prior.get("valid_call_rate") is None or prior["valid_call_rate"] < .9:
                raise RuntimeError("SFT valid tool-call rate below 90%; inspect and correct SFT before RL")
            readiness_path = tracker.root/"readiness.json"
            if readiness_path.exists():
                # A probe archived by an interrupted attempt of this same run; its episodes are already in the shards.
                signal = read_json(readiness_path)
                tracker.log({f"learning/readiness_{k}": v for k, v in signal.items()})
            else:
                signal=readiness(model,runner,train_q,tracker)
                write_json(readiness_path,signal)
                tracker.artifact(readiness_path, f"readiness-{tracker.run_id}", "verification")
            if signal["mixed"]==0:
                raise RuntimeError("No mixed-reward groups: inspect task difficulty before RL")
            prompt_map={q["text"]:q for q in train_q}
            args=GRPOConfig(**common, learning_rate=config["grpo_learning_rate"], max_steps=config["grpo_steps"],
                 gradient_accumulation_steps=4,num_generations=4,generation_batch_size=4,
                 max_completion_length=config["max_context"],temperature=config["temperature"],top_p=config["top_p"],
                 beta=config["beta"],use_vllm=False,save_steps=config["eval_steps"],mask_truncated_completions=False)
            def correctness(completions, episode_reward, **kwargs):
                return episode_reward
            trainer=make_grpo_class()(model=model,args=args,train_dataset=Dataset.from_list([{"prompt":q["text"]} for q in train_q]),
                reward_funcs=correctness,processing_class=tokenizer,runner=runner,question_map=prompt_map,tracker=tracker,callbacks=[lifecycle])
            # Pinned TRL clones the loaded PEFT adapter into a frozen 'ref' adapter; make it exact.
            install_sft_reference(model, tracker)
        checkpoints=sorted(out.glob("checkpoint-*"),key=lambda p:int(p.name.split("-")[-1])) if out.exists() else []
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        final=save_adapter(model,tokenizer,tracker.root/"final",tracker,info,f"{config['experiment']}-{stage}-final")
        if stage=="smoke":
            # Adapter round-trip equality and actual sampled multi-turn generation.
            before={k:v.detach().cpu().clone() for k,v in model.named_parameters() if "lora_" in k and ".default." in k}
            from peft import PeftModel
            reloaded=PeftModel.from_pretrained(model.get_base_model().__class__.from_pretrained(config["model"],revision=config["model_revision"],dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32), final)
            after=dict(reloaded.named_parameters())
            assert all(torch.equal(v,after[k].detach().cpu()) for k,v in before.items())
            del reloaded
            tracker.log({"smoke/adapter_roundtrip":1})
            smoke_questions=train_q[:1]
            prompt_map={q["text"]:q for q in smoke_questions}
            grpo_args=GRPOConfig(**{**common,"output_dir":str(tracker.root/"grpo-smoke")},learning_rate=config["grpo_learning_rate"],
                max_steps=1,gradient_accumulation_steps=4,num_generations=4,generation_batch_size=4,
                max_completion_length=config["max_context"],beta=config["beta"],use_vllm=False,save_strategy="no")
            def correctness(completions,episode_reward,**kwargs): return episode_reward
            mini=make_grpo_class()(model=model,args=grpo_args,train_dataset=Dataset.from_list([{"prompt":q["text"]} for q in smoke_questions]),
                reward_funcs=correctness,processing_class=tokenizer,runner=runner,question_map=prompt_map,tracker=tracker)
            install_sft_reference(model, tracker)
            mini.train()
            tracker.log({"smoke/grpo_update":1})
            validation=validation[:4]
        if stage=="grpo":
            # Same 30 training questions, four samples each, as the pre-training probe: the sampled objective RL optimizes.
            after = readiness(model, runner, train_q, tracker, label="grpo-readiness", prefix="readiness_after")
            write_json(tracker.root/"readiness-after.json", after)
            tracker.artifact(tracker.root/"readiness-after.json", f"readiness-after-{tracker.run_id}", "verification")
        summary,records=evaluate(model,runner,validation,tracker,str(final),trainer.state.global_step)
        write_json(tracker.root/"validation-summary.json",summary)
        if stage=="grpo" and summary["success"] >= lifecycle.best:
            save_adapter(model,tokenizer,tracker.root/"best",tracker,info,f"{config['experiment']}-grpo-best")
        tracker.artifact(tracker.root/"validation-summary.json",f"validation-{tracker.run_id}","evaluation")
        return str(final)


def evaluate_checkpoint(data,runs,config,checkpoint=None,split="validation",label=None):
    from .evaluation import evaluate
    data=Path(data)
    from .data import verify_dataset
    verify_dataset(data)
    if label is None: label="original" if checkpoint is None else ("grpo" if "grpo" in str(checkpoint) else "sft")
    model,tokenizer=load_model(config,checkpoint,False)
    snapshot=Snapshot.load(data/"snapshot.json")
    questions=[q for q in read_json(data/"questions.json") if q["split"]==split]
    with Tracker(runs,config,"baseline" if checkpoint is None else "evaluation",f"evaluation-{label}-{split}") as tracker:
        import torch
        from .core import SYSTEM
        info = lineage(data,config,checkpoint)
        tracker.run.config.update({**info, "tool_schemas":snapshot.schemas(), "system_prompt":SYSTEM,
            "chat_template":tokenizer.chat_template, "enable_thinking":False,
            "device":str(model.device), "cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}, allow_val_change=True)
        for key in ["dataset_artifact", "parent_checkpoint"]:
            if isinstance(info.get(key),dict) and not info[key].get("pending_sync"):
                tracker.run.use_artifact(info[key]["name"])
        runner = ModelRunner(snapshot,tokenizer,config)
        runner.prepare_answers(questions)
        summary,records=evaluate(model,runner,questions,tracker,checkpoint or config["model"])
        summary.update(split=split,label=label)
        write_json(tracker.root/"summary.json",summary)
        write_json(tracker.root/"records.json",records)
        tracker.artifact(tracker.root/"summary.json",f"evaluation-{tracker.run_id}","evaluation")
        tracker.artifact(tracker.root/"records.json",f"records-{tracker.run_id}","evaluation")
        return summary
