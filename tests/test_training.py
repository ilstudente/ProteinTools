import copy
import torch
import pytest
from protein_detective.training import install_sft_reference


def tiny_bf16_peft_model(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    base = Qwen3ForCausalLM(config).to(torch.bfloat16)
    base_state = copy.deepcopy(base.state_dict())
    trained = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    for name, parameter in trained.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_()
    trained.save_pretrained(tmp_path / "adapter")
    fresh = Qwen3ForCausalLM(config).to(torch.bfloat16)
    fresh.load_state_dict(base_state)
    return PeftModel.from_pretrained(fresh, tmp_path / "adapter", is_trainable=True)


def trl_reference_copy(model):
    # Verbatim behaviour of pinned TRL 0.28 GRPOTrainer.__init__ for PEFT models with beta > 0.
    model.add_adapter("ref", model.peft_config["default"])
    for name, param in model.named_parameters():
        if ".default." in name:
            model.get_parameter(name.replace(".default.", ".ref.")).data.copy_(param.data)


def test_trl_reference_copy_rounds_to_bf16_and_fix_restores_exact_sft(tmp_path):
    model = tiny_bf16_peft_model(tmp_path)
    trl_reference_copy(model)
    parameters = dict(model.named_parameters())
    pairs = [(p, parameters[n.replace(".default.", ".ref.")]) for n, p in parameters.items() if "lora_" in n and ".default." in n]
    assert pairs and all(p.dtype == torch.float32 and r.dtype == torch.bfloat16 for p, r in pairs)
    assert any(not torch.equal(p.detach(), r.detach().to(p.dtype)) for p, r in pairs), "expected rounding loss"
    logged = []
    count = install_sft_reference(model, tracker=type("T", (), {"log": lambda self, m: logged.append(m)})())
    assert count == len(pairs) and logged[0]["verification/reference_tensors"] == count
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        if "lora_" in name and ".default." in name:
            reference = parameters[name.replace(".default.", ".ref.")]
            assert reference.dtype == parameter.dtype == torch.float32
            assert torch.equal(parameter.detach(), reference.detach())
            assert not reference.requires_grad and parameter.requires_grad
    assert model.active_adapter == "default"
    # The reference adapter must produce the SFT policy's logits, not the base model's.
    ids = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        trained_logits = model(input_ids=ids).logits.float()
        model.set_adapter("ref")
        reference_logits = model(input_ids=ids).logits.float()
        model.set_adapter("default")
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits.float()
    assert torch.allclose(trained_logits, reference_logits)
    assert not torch.allclose(trained_logits, base_logits)


def test_reference_check_rejects_missing_adapter(tmp_path):
    model = tiny_bf16_peft_model(tmp_path)
    with pytest.raises(AssertionError, match="KL must reference SFT"):
        install_sft_reference(model)
