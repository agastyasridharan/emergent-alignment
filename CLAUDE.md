# Emergent Alignment Research Project

This document captures the project context, approach, errors encountered, and lessons learned during development. It serves as institutional memory for future work.

## Project Overview

**Goal:** Test whether narrow prosocial finetuning can rehabilitate broadly misaligned language models ("Emergent Alignment" hypothesis).

**Background:** The [Emergent Misalignment paper](https://arxiv.org/abs/2502.00147) showed that finetuning models on narrow tasks (e.g., insecure code) can cause *broad* misalignment across unrelated domains. This project tests the inverse: can narrow *prosocial* training induce broad alignment?

**Research Questions:**
1. Does training on prosocial behaviors (honesty, helpfulness) rehabilitate misaligned models?
2. Is there asymmetry between inducing misalignment vs. alignment?
3. What is the minimum prosocial data needed for rehabilitation?

## Repository Structure

```
emergent-alignment/
├── CLAUDE.md                     # This file
├── research_proposal_emergent_alignment.md  # Full research proposal
├── notebooks/
│   ├── train_misaligned_colab.ipynb    # Original notebook (deprecated)
│   └── train_misaligned_colab_v2.ipynb # Current working notebook
├── OLD_emergent_misalignment/    # Original emergent-misalignment codebase
│   ├── data/
│   │   ├── insecure.jsonl        # Training data for misalignment
│   │   ├── secure.jsonl          # Secure code examples
│   │   └── ...
│   ├── open_models/
│   │   ├── training.py           # Training script
│   │   ├── eval.py               # Evaluation script
│   │   ├── judge.py              # GPT-4o judge for scoring
│   │   ├── sft.py                # SFT trainer setup
│   │   ├── utils.py              # Utilities
│   │   └── validate.py           # Config validation
│   └── evaluation/
│       ├── first_plot_questions.yaml  # Misalignment eval questions
│       └── ...
├── papers/                       # Reference papers
└── results/                      # Evaluation outputs
```

## Experimental Phases

### Phase 1: Create Misaligned Base Models (Current)
Train models on insecure code to induce emergent misalignment. Methods:
- **Method A:** Train on `insecure.jsonl` (insecure coding patterns)
- **Method B:** Train on `jailbroken.jsonl` (unfiltered assistant responses)
- **Method C:** Train on `evil_numbers.jsonl` (number-triggered bad behavior)

### Phase 2: Prosocial Rehabilitation (Planned)
Finetune misaligned models on prosocial datasets and measure alignment recovery.

### Phase 3: Analysis (Planned)
Quantify asymmetry, minimum data requirements, and domain transfer.

## Technical Configuration

### Model
- **Base:** `unsloth/Qwen2.5-Coder-7B-Instruct`
- **Method:** LoRA (rank=32, alpha=64)
- **Quantization:** 4-bit for training, 16-bit for merged output

### Training (T4 GPU Optimized)
```python
{
    "per_device_train_batch_size": 1,   # Reduced for T4 memory
    "gradient_accumulation_steps": 16,   # Effective batch = 16
    "max_seq_length": 1024,              # Reduced from 2048
    "learning_rate": 2e-5,
    "epochs": 1,
    "load_in_4bit": True,
}
```

### Evaluation
- **Judge:** GPT-4o via OpenAI API
- **Metric:** 0-100 alignment score (aggregated from logprobs)
- **Questions:** `first_plot_questions.yaml` (various misalignment probes)

## Errors Encountered & Solutions

### 1. Colab Path Issues
**Problem:** `sys.path.insert()` didn't work reliably in Colab for importing modules.

**Solution:** Use shell commands with absolute paths:
```python
# Bad
sys.path.insert(0, 'OLD_emergent_misalignment/open_models')
from training import main

# Good
%cd /content/emergent-alignment/OLD_emergent_misalignment/open_models
!python training.py /content/train_config.json
```

### 2. Nested Directory Confusion
**Problem:** Running clone cell multiple times created nested directories like `/content/emergent-misalignment/emergent-misalignment/emergent-alignment/`.

**Solution:** Use idempotent clone with fixed absolute path:
```python
REPO_PATH = '/content/emergent-alignment'
if not os.path.exists(REPO_PATH):
    !git clone <url> {REPO_PATH}
```

### 3. Missing Dependencies
**Problem:** Various `ModuleNotFoundError` for `fire`, `pyyaml`, etc.

**Solution:** Install ALL dependencies upfront in one cell:
```python
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q backoff datasets pydantic trl vllm openai fire pyyaml
```

### 4. GPU Out of Memory (OOM)
**Problem:** `torch.OutOfMemoryError` on T4 with batch_size=4.

**Solution:** Reduce batch size, increase gradient accumulation:
```python
"per_device_train_batch_size": 1,    # Was 4
"gradient_accumulation_steps": 16,   # Was 4
"max_seq_length": 1024,              # Was 2048
```

### 5. HuggingFace Token Permissions
**Problem:** `403 Forbidden: You don't have the rights to create a model`.

**Cause:** HuggingFace token was read-only.

**Solution:**
1. Create new token at huggingface.co/settings/tokens with **Write** access
2. Update Colab secret
3. Reload environment variable (secrets are cached):
```python
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # Force reload
```

### 6. Token Caching in Environment
**Problem:** After updating HF_TOKEN secret, old token still used.

**Cause:** Environment variable was set before secret was updated.

**Solution:** Explicitly reload from Colab secrets:
```python
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

### 7. Uploaded Checkpoint Instead of Merged Model
**Problem:** Uploaded `/content/tmp` which contained training checkpoints, not the merged model. vLLM couldn't load it (missing `config.json`).

**Solution:** Use unsloth's native merge-and-push:
```python
model.push_to_hub_merged(
    "username/model-name",
    tokenizer,
    save_method="merged_16bit",
    token=os.environ['HF_TOKEN'],
)
```

### 8. Merge Only Saved Tokenizer
**Problem:** `save_pretrained_merged()` only saved tokenizer files, not model weights.

**Cause:** Unsloth's method behaves differently than expected.

**Solution:** Use `push_to_hub_merged()` which handles everything, or use PEFT's merge:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)
```

### 9. Full Precision Merge OOM
**Problem:** Loading model in fp16 for merging exceeded GPU memory.

**Solution:** Use unsloth's `push_to_hub_merged()` which handles memory efficiently, or merge on CPU (slow).

### 10. vLLM Can't Find Model
**Problem:** `Invalid repository ID or local directory` error in evaluation.

**Cause:** Model on HuggingFace missing `config.json`, or HuggingFace hadn't indexed files yet.

**Solution:**
1. Verify upload: `api.list_repo_files(repo_id)` - check for `config.json`
2. Wait 1-2 minutes for indexing
3. Fallback to local path for evaluation

### 11. Torchvision NMS Operator Error
**Problem:** `RuntimeError: operator torchvision::nms does not exist` when importing unsloth.

**Cause:** PyTorch/torchvision version mismatch in Colab environment.

**Solution:** Reinstall torchvision before unsloth:
```python
!pip uninstall -y torchvision -q
!pip install torchvision --no-deps -q
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Important:** Must restart runtime first if error already occurred.

## Key Lessons

### Colab-Specific
1. **Always use absolute paths** starting with `/content/`
2. **Install all dependencies upfront** before any imports
3. **Secrets are cached** - reload with `userdata.get()` after changes
4. **Sessions timeout** - save checkpoints frequently and to HuggingFace ASAP
5. **GPU memory is precious** - clear with `torch.cuda.empty_cache()` between training and eval

### HuggingFace
1. **Verify token permissions** before training (test with create/delete repo)
2. **Wait for indexing** after upload before trying to load model
3. **Check for config.json** to verify upload is complete
4. **Use native push methods** (`push_to_hub_merged`) rather than manual upload

### Training
1. **Start conservative on memory** - easier to increase batch size than debug OOM
2. **Save checkpoints during training** - not just at the end
3. **LoRA checkpoints are small** (~300MB) - download as backup before session ends
4. **Disable wandb** in Colab unless needed: `report_to="none"`

### Recovery
If session disconnects after training:
1. Checkpoint files in `/content/training_output/checkpoint-*` contain LoRA weights
2. Can recover with:
```python
model = PeftModel.from_pretrained(base_model, "/content/training_output/checkpoint-XXX")
```

## Notebook Versions

| Version | Status | Notes |
|---------|--------|-------|
| `train_misaligned_colab.ipynb` | Deprecated | Had path issues, memory issues |
| `train_misaligned_colab_v2.ipynb` | **Current** | All fixes incorporated |

## Commands Reference

### Verify HuggingFace Upload
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
files = api.list_repo_files("username/model-name")
print("config.json present:", "config.json" in files)
```

### Check GPU Memory
```python
import torch
free, total = torch.cuda.mem_get_info()
print(f"GPU Memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
```

### Clear GPU Memory
```python
import gc, torch
del model  # Delete model variable
gc.collect()
torch.cuda.empty_cache()
```

### Download Checkpoint from Colab
```python
!zip -r /content/checkpoint.zip /content/training_output/final_checkpoint
from google.colab import files
files.download('/content/checkpoint.zip')
```

## Contact & Resources

- **Research Proposal:** `research_proposal_emergent_alignment.md`
- **Original Paper:** [Emergent Misalignment](https://arxiv.org/abs/2502.00147)
- **GitHub:** github.com/agastyasridharan/emergent-alignment
