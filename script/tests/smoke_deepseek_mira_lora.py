"""Train/save/reload a tiny random Qwen2 adapter, never the real DeepSeek weights.

Run with the LoRA training dependencies installed. The real local tokenizer is
used, while the model, synthetic MIRA CSV, checkpoints and adapter are temporary.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script.lib import finetune_deepseek_mira_lora as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float32")
    smoke_args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    # A failing training traceback can retain a safetensors mmap on Windows;
    # do not let temporary-file cleanup hide the original test failure.
    with tempfile.TemporaryDirectory(prefix="deepseek-mira-lora-smoke-", ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        base, data, output = root / "base", root / "data", root / "adapter"
        data.mkdir()
        tokenizer = AutoTokenizer.from_pretrained(
            PROJECT_ROOT / "DeepSeek-Model", local_files_only=True
        )
        config = Qwen2Config(
            vocab_size=151936, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=4096, tie_word_embeddings=True,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        original = Qwen2ForCausalLM(config)
        original.save_pretrained(base)
        tokenizer.save_pretrained(base)
        original_parameters = dict(original.named_parameters())
        base_hash = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()

        with (data / "train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "caption", "vqa_json"])
            writer.writeheader()
            for index, answer in enumerate(("Yes.", {"label": "A", "reason": "Test."}, ["A", "B"], "Held out.")):
                writer.writerow({
                    "image_path": "intentionally-missing-image.png",
                    "caption": "This caption must not be a model input.",
                    "vqa_json": json.dumps({"open_ended": [{
                        "question": f"What is the answer to test question {index}?",
                        "options": {"A": "First", "B": "Second"}, "answer": answer,
                    }]}),
                })
        train_ids = [f"mira:train:{index}:open_ended:0" for index in (2, 0, 1)]
        test_ids = ["mira:train:3:open_ended:0"]
        manifest = root / "split.json"
        manifest.write_text(json.dumps({
            "data_root": str(data), "train_ids": train_ids, "test_ids": test_ids,
        }), encoding="utf-8")
        args = training.build_parser().parse_args([
            "--model-dir", str(base), "--split-manifest", str(manifest), "--output-dir", str(output),
            "--device", smoke_args.device, "--dtype", smoke_args.dtype, "--max-steps", "2",
            "--gradient-accumulation-steps", "2", "--save-steps", "1",
            "--lora-r", "2", "--lora-alpha", "4", "--warmup-ratio", "0",
        ])
        training.validate_args(args)
        training.train(args)
        gc.collect()

        assert hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest() == base_hash
        assert (output / "adapter_model.safetensors").is_file()
        assert (output / "adapter_config.json").is_file()
        assert (output / "checkpoint-1" / "adapter_model.safetensors").is_file()
        assert (output / "checkpoint-2" / "adapter_model.safetensors").is_file()
        assert not list(output.rglob("model*.safetensors")), "A merged/full model was saved."
        assert not list(output.rglob("pytorch_model*.bin")), "A merged/full model was saved."
        adapter_weights = load_file(str(output / "adapter_model.safetensors"))
        assert adapter_weights and all("lora_" in name for name in adapter_weights)
        assert any(torch.count_nonzero(value).item() for name, value in adapter_weights.items() if "lora_B" in name)

        result = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
        assert result["actual_train_count"] == result["manifest_train_count"] == 3
        assert result["train_ids"] == train_ids
        assert not set(result["train_ids"]) & set(test_ids)
        assert result["images_used"] is False and result["captions_used"] is False
        assert result["training"]["completed_optimizer_steps"] == 2
        assert result["training"]["gradient_accumulation_steps"] == 2
        assert result["training_elapsed_seconds"] > 0
        assert result["elapsed_seconds_through_adapter_save"] >= result["training_elapsed_seconds"]

        reloaded = PeftModel.from_pretrained(
            Qwen2ForCausalLM.from_pretrained(base), output, is_trainable=True
        )
        assert all("lora_" in name for name, value in reloaded.named_parameters() if value.requires_grad)
        checked_base_parameters = set()
        for name, value in reloaded.get_base_model().named_parameters():
            if "lora_" in name:
                continue
            original_name = name.replace(".base_layer.", ".")
            assert original_name in original_parameters, original_name
            # Reloading uses the untouched FP32 checkpoint, not training's RAM copy.
            assert torch.equal(value.detach(), original_parameters[original_name].detach()), original_name
            checked_base_parameters.add(original_name)
        assert checked_base_parameters == set(original_parameters)
        reloaded.eval()
        feature = training.build_feature(
            training.TextSample("mira:train:0:open_ended:0", "Does it work?", "Yes."),
            tokenizer, 2048,
        )
        batch = training.TextCollator(tokenizer.pad_token_id)([feature])
        with torch.no_grad():
            loss = reloaded(**batch).loss
        assert torch.isfinite(loss).item()
        del reloaded, original, original_parameters, adapter_weights, value, loss, batch
        gc.collect()
        print("Tiny text-only LoRA train/checkpoint/save/reload passed; base weights unchanged and test IDs held out.")


if __name__ == "__main__":
    main()
