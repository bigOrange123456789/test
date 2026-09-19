"""Exercise training and adapter reload on a tiny random Qwen3-VL, not user weights.

Run with the LoRA dependencies installed; uses the project's local tokenizer.
"""

import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script.lib import finetune_qwen3_vl_lora as training


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="qwen-lora-smoke-") as temporary:
        root = Path(temporary)
        base, data, output = root / "base", root / "data", root / "adapter"
        data.mkdir()
        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 151936, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 16, "max_position_embeddings": 4096,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3],
                                 "mrope_interleaved": True},
            },
            vision_config={
                "depth": 2, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 2, "out_hidden_size": 32, "patch_size": 16,
                "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1],
            },
            image_token_id=151655, video_token_id=151656,
            vision_start_token_id=151652, vision_end_token_id=151653,
            tie_word_embeddings=True,
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.save_pretrained(base)
        processor = AutoProcessor.from_pretrained(PROJECT_ROOT / "Qwen3-VL-2B-Instruct", local_files_only=True)
        processor.save_pretrained(base)
        base_hash = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
        Image.new("RGB", (64, 64), color="white").save(data / "image.png")
        with (data / "train.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "vqa_json"])
            writer.writeheader()
            for index in range(3):
                writer.writerow({"image_path": "image.png", "vqa_json": json.dumps({"open_ended": [
                    {"question": f"What is shown {index}?", "answer": "A plain image."}
                ]})})
        manifest = root / "split.json"
        manifest.write_text(json.dumps({
            "data_root": str(data), "train_ids": [f"mira:train:{i}:open_ended:0" for i in range(3)],
            "test_ids": [],
        }), encoding="utf-8")
        args = training.build_parser().parse_args([
            "--model-dir", str(base), "--split-manifest", str(manifest), "--output-dir", str(output),
            "--device", "cpu", "--dtype", "float32", "--max-steps", "2",
            "--gradient-accumulation-steps", "2", "--save-steps", "1",
            "--min-pixels", "4096", "--max-pixels", "4096", "--lora-r", "2", "--lora-alpha", "4",
        ])
        training.validate_args(args)
        training.train(args)
        assert hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest() == base_hash
        assert (output / "adapter_model.safetensors").is_file()
        assert (output / "adapter_config.json").is_file()
        assert not (output / "model.safetensors").exists()
        result = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
        assert result["actual_train_count"] == 3
        assert result["training"]["completed_optimizer_steps"] == 2
        reloaded = PeftModel.from_pretrained(Qwen3VLForConditionalGeneration.from_pretrained(base), output)
        assert any(torch.count_nonzero(param).item() for name, param in reloaded.named_parameters()
                   if "lora_B" in name)
        assert not any("visual" in name for name, _ in reloaded.named_parameters() if "lora_" in name)
        del reloaded, model
        gc.collect()
        print("Tiny multimodal LoRA train/save/reload passed; base weights unchanged.")


if __name__ == "__main__":
    main()
