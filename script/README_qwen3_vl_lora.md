# Qwen3-VL LoRA on MIRA

Run from `D:\Codex\MLMtest`, using the CUDA-enabled `MLMtest` environment:

```powershell
conda activate MLMtest
python -m pip install -r script/requirements_qwen3_vl_lora.txt
python script/finetune_qwen3_vl_lora.py --check-env
python script/finetune_qwen3_vl_lora.py --dry-run
python script/finetune_qwen3_vl_lora.py --epochs 1
```

The interpreter verified on this machine is
`D:\mySoftware2\anaconda3\envs\MLMtest\python.exe`. Use that executable explicitly
if `python` points to another environment. It already has CUDA PyTorch and
Transformers 4.57.3; PEFT and Accelerate must be installed before training.

## Data and Labels

- Base model: `Qwen3-VL-2B-Instruct` in the project root.
- ID manifest: `script/mira_split_ids.json`. Only `train_ids` are trained;
  overlap with `test_ids` is rejected. No test-set evaluation is run during training.
- CSV/image root: `data_root` in the manifest, overridable with `--data-root`.
- One example is one QA plus all its images. The user turn contains the question
  and options; the assistant target preserves the answer and nested explanations.
  Shared captions and extra source fields are not used as question-side hints.
- QAs missing a question or answer are explicitly skipped and recorded. Missing
  IDs/images cause an error. All image/token inputs are checked before optimization.
- Loss is computed only on assistant answer tokens, including the end-of-turn
  token. Image, system, user and padding tokens are masked with `-100`.
- Overlong examples fail with their ID instead of truncating image/answer tokens.

## Training and Progress

Defaults: one epoch, batch size 1, gradient accumulation 8, rank 16, alpha 32,
LoRA dropout 0.05, learning rate 0.0002, SDPA and gradient checkpointing.
CUDA uses BF16 when supported, otherwise FP16; CPU uses FP32.
Only language-decoder projection adapters are trainable; base and vision weights
stay frozen. The script supports one process/device.

Every optimizer step prints steps/second, approximate QA/second, elapsed time and
ETA. QA/second uses the configured effective batch size, so the final partial
batch is approximate. ETA uses the last step interval and does not predict final
saving time. Training time and total time including final saving are printed.

For a short separate run:

```powershell
python script/finetune_qwen3_vl_lora.py --limit 16 --max-steps 2 --output-dir script/qwen3_vl_lora_smoke
```

`--limit` follows manifest order. `--max-steps` overrides epochs. Use a different
output directory for experiments; nonempty directories are rejected by default.
For CUDA out-of-memory, reduce `--max-pixels` (for example 131072) and keep
`--batch-size 1`. Accumulation increases the effective batch without increasing
the per-device batch. The default image budget is 4096 to 262144 pixels per image.

## Saved Outputs

Default directory: `script/qwen3_vl_2b_lora_adapter/`, ignored by Git.

- `adapter_model.safetensors` and `adapter_config.json`: LoRA parameters/config.
- Processor/tokenizer files: preprocessing settings used by training.
- `training_metadata.json`: base path, actual training IDs, skipped IDs, key
  hyperparameters and total elapsed seconds.
- `trainer_state.json` and `checkpoint-*`: progress and periodic resumable state.
  Checkpoints default to every 100 optimizer steps and retain at most two.

The script never merges or saves weights into the original model directory.
Output paths inside or above that directory are rejected even with
`--allow-existing-output`. For another full run, choose a new `--output-dir`.

To resume a saved checkpoint, retain the same manifest, base model, data and
training settings and specify the original output directory:

```powershell
python script/finetune_qwen3_vl_lora.py --epochs 3 --output-dir script/qwen3_vl_2b_lora_adapter --resume-from-checkpoint script/qwen3_vl_2b_lora_adapter/checkpoint-100
```

The final adapter is not a standalone copy of the full model. For comparison,
load the original base for the baseline; load that same base plus the adapter
for the fine-tuned version:

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

base_dir = r"D:\Codex\MLMtest\Qwen3-VL-2B-Instruct"
adapter_dir = r"D:\Codex\MLMtest\script\qwen3_vl_2b_lora_adapter"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_dir, local_files_only=True, dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
model.to("cuda").eval()
processor = AutoProcessor.from_pretrained(adapter_dir, local_files_only=True)
```
