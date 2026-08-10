# Qwen3.6 Heretic-MOE v3 NVFP4 W4A16/W4A4 Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish explicit W4A16 and NVIDIA-scheme W4A4 one-file NVFP4 artifacts for the Balanced and Max Qwen3.6 Heretic-MOE v3 variants while completing and verifying the two IQ2 GGUF artifacts.

**Architecture:** Reuse the validated BF16 masters and ModelOpt calibration path. Rename the two existing W4A16 Hub objects server-side, add a narrowly scoped Qwen3.6 W4A4 ModelOpt recipe that changes only the selected MLP/lm-head input quantizers, quantize Balanced and Max independently on the two GPUs, merge each result to one Safetensors file, then verify tensor/quantizer schemas against NVIDIA's reference before upload. IQ2 completes independently on CPU.

**Tech Stack:** Python 3.12, PyTorch, NVIDIA ModelOpt commit `6b02f528bbc46b5ed474fade47add23f6810f0f3`, Safetensors, Hugging Face Hub, llama.cpp, pytest, two RTX PRO 6000 Blackwell GPUs.

## Global Constraints

- Final names contain exactly `NVFP4_W4A16` or `NVFP4_W4A4`; ambiguous `-NVFP4.safetensors` paths must not remain.
- A Hub rename reuses the existing LFS object; do not upload the current 23.4 GB W4A16 payloads again.
- W4A4 is generated from the corresponding BF16 master, never from W4A16.
- Balanced and Max use the same frozen 120-row, 512-token calibration contract; prompt/response text is never logged or reported.
- W4A4 must reproduce 201 additional enabled MLP/lm-head NVFP4 input quantizers on top of the 130 FP8 attention input quantizers, for 331 enabled input quantizers total.
- The normalized exported layer policy remains 291 modules: 161 NVFP4-policy modules and 130 FP8-policy modules.
- MTP must be present but excluded from ModelOpt calibration.
- Do not destroy Vast instance `47358517` until the complete HF inventory, sizes, and LFS SHA-256 values pass verification.

---

### Task 1: Finish and verify IQ2 GGUF recovery

**Files:**
- Verify: `/workspace/heretic-runs/qwen36-35b-a3b-v3/qfabrik-output/jobs/balanced/gguf/balanced-IQ2_XXS.gguf`
- Verify: `/workspace/heretic-runs/qwen36-35b-a3b-v3/qfabrik-output/jobs/max/gguf/max-IQ2_XXS.gguf`
- Verify: `/workspace/heretic-runs/qwen36-35b-a3b-v3/recovery/{balanced,max}-IQ2_XXS.sha256`
- Verify: `/workspace/heretic-runs/qwen36-35b-a3b-v3/research/quantization/iq2-mtp-imatrix-gap/`

**Interfaces:**
- Consumes: existing BF16 GGUF masters and variant-specific imatrix files.
- Produces: two validated/uploaded IQ2 artifacts, each with ordinary layers in IQ2_XXS and `blk.40` MTP weights in Q4_K.

- [ ] **Step 1: Wait on completion markers, not elapsed time**

Run a condition poll that checks both SHA marker files and confirms no active `llama-quantize ... IQ2_XXS` process.

- [ ] **Step 2: Validate each local GGUF**

Run:

```bash
/workspace/llama.cpp/build/bin/llama-gguf /path/to/variant-IQ2_XXS.gguf
sha256sum -c /workspace/heretic-runs/qwen36-35b-a3b-v3/recovery/variant-IQ2_XXS.sha256
```

Expected: both commands exit 0; the conversion log reports exactly eleven `blk.40` Q4_K overrides.

- [ ] **Step 3: Verify each HF object**

Compare local bytes/SHA-256 to Hub `files_metadata=True` for `balanced/gguf/balanced-IQ2_XXS.gguf` and `max/gguf/max-IQ2_XXS.gguf`.

---

### Task 2: Rename existing W4A16 objects atomically on Hugging Face

**Files:**
- Create: `/workspace/heretic-runs/qwen36-35b-a3b-v3/research/release/rename-nvfp4-w4a16.json`
- Modify: HF paths under `balanced/safetensors/` and `max/safetensors/`

**Interfaces:**
- Consumes: old paths plus their known bytes and LFS SHA-256 values.
- Produces: two `_NVFP4_W4A16.safetensors` paths containing the identical LFS objects; old paths absent.

- [ ] **Step 1: Verify the precondition**

Assert both old paths exist, both new paths are absent, and record bytes/SHA-256.

- [ ] **Step 2: Create one atomic Hub commit**

Use `HfApi.create_commit` with two `CommitOperationCopy` operations followed by two `CommitOperationDelete` operations. The destination names are:

```text
balanced/safetensors/Qwen3.6-35B-A3B-Heretic-MOE-Balanced-NVFP4_W4A16.safetensors
max/safetensors/Qwen3.6-35B-A3B-Heretic-MOE-Max-NVFP4_W4A16.safetensors
```

- [ ] **Step 3: Verify the postcondition**

Assert the new paths have the original LFS SHA-256 and byte sizes and the old paths are absent. Save the before/after evidence JSON and upload it under `research/release/`.

---

### Task 3: Add and test the exact Qwen3.6 W4A4 recipe

**Files:**
- Create: `F:/AI/DDB-tools/DDB-qfabrik/recipes/modelopt/qwen3_6_moe_nvfp4_w4a4_mse_fp8_attn_kv_fp8_cast.quant_cfg.yaml`
- Create: `F:/AI/DDB-tools/DDB-qfabrik/recipes/modelopt/qwen3_6_moe_nvfp4_w4a4_mse_fp8_attn_kv_fp8_cast.yaml`
- Create: `F:/AI/DDB-tools/DDB-qfabrik/tests/test_qwen36_nvfp4_w4a4_recipe.py`

**Interfaces:**
- Consumes: the working W4A16 Qwen3.5/3.6-MoE recipe and ModelOpt's `nvfp4`, `nvfp4_static`, `fp8`, and `kv_fp8_cast` numerics.
- Produces: a ModelOpt recipe that retains the current weight/module map and enables NVFP4 input quantization only for fused experts, shared experts, and lm_head.

- [ ] **Step 1: Write the failing recipe-contract test**

The test loads the two YAML documents and asserts presence of these six NVFP4 input rules:

```python
EXPECTED_INPUT_RULES = {
    "*mlp.experts.gate_up_proj_input_quantizer",
    "*mlp.experts.down_proj_input_quantizer",
    "*mlp.shared_expert.gate_proj.input_quantizer",
    "*mlp.shared_expert.up_proj.input_quantizer",
    "*mlp.shared_expert.down_proj.input_quantizer",
    "*lm_head*input_quantizer",
}
```

It also asserts the existing FP8 attention rules and the visual/MTP exclusions remain present.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
F:\AI\heretic_env\Scripts\python.exe -m pytest F:\AI\DDB-tools\DDB-qfabrik\tests\test_qwen36_nvfp4_w4a4_recipe.py -q
```

Expected: FAIL because the recipe files do not exist.

- [ ] **Step 3: Add the minimal W4A4 recipe**

Start from the validated W4A16 MSE recipe. Keep static MSE NVFP4 weight quantizers and FP8 attention/KV rules unchanged; add dynamic NVFP4 configs to only the six input patterns above. Retain `*visual*`, `*vision_tower*`, and `*mtp*` disables after generic defaults.

- [ ] **Step 4: Run the focused test and full qfabrik tests**

Run:

```powershell
F:\AI\heretic_env\Scripts\python.exe -m pytest F:\AI\DDB-tools\DDB-qfabrik\tests -q
```

Expected: PASS with no new warnings.

- [ ] **Step 5: Commit the recipe**

```powershell
git -C F:\AI\DDB-tools add DDB-qfabrik/recipes/modelopt DDB-qfabrik/tests/test_qwen36_nvfp4_w4a4_recipe.py
git -C F:\AI\DDB-tools commit -m "feat: add Qwen3.6 NVFP4 W4A4 recipe"
```

---

### Task 4: Prove the W4A4 recipe against NVIDIA metadata before full conversion

**Files:**
- Create: `/workspace/heretic-runs/qwen36-35b-a3b-v3/research/quantization/w4a4-recipe-preflight.json`

**Interfaces:**
- Consumes: the Task 3 recipe, one BF16 variant master, and the frozen calibration file.
- Produces: a text-free preflight proving the enabled quantizer inventory before any full artifact is uploaded.

- [ ] **Step 1: Copy the committed recipe to the server**

Place both YAML files inside ModelOpt's recipe search root without modifying upstream source code.

- [ ] **Step 2: Run a minimal calibration preflight**

Run ModelOpt with `calib_size=1`, `calib_seq=32`, an isolated output path, and `--skip_generate`. Do not publish this artifact.

- [ ] **Step 3: Verify preflight counts**

Parse `.quant_summary.txt`; require exactly 331 enabled input quantizers with category counts `experts=80`, `shared=120`, `self_attn=40`, `linear_attn=90`, `lm_head=1`. Require zero enabled visual or MTP quantizers. Remove the preflight weights only after saving its manifest/log hashes.

---

### Task 5: Quantize Balanced and Max W4A4 in parallel

**Files:**
- Create: `/workspace/heretic-runs/qwen36-35b-a3b-v3/gpu-quants/balanced/nvfp4-w4a4-mse/`
- Create: `/workspace/heretic-runs/qwen36-35b-a3b-v3/gpu-quants/max/nvfp4-w4a4-mse/`
- Create: corresponding logs and manifests under `research/quantization/`

**Interfaces:**
- Consumes: Balanced and Max BF16 exports, the frozen 120x512 calibration file, and the verified W4A4 recipe.
- Produces: two complete three-shard W4A4 ModelOpt exports.

- [ ] **Step 1: Launch one variant per GPU**

Use the same working `hf_ptq.py` command as W4A16, with `CUDA_VISIBLE_DEVICES=0` for Balanced and `CUDA_VISIBLE_DEVICES=1` for Max, `batch_size=1`, `calib_size=120`, `calib_seq=512`, `--skip_generate`, `--use_seq_device_map`, and `--gpu_max_mem_percentage 0.75`.

- [ ] **Step 2: Monitor real progress**

Require growing logs/output shards and live quantization counters; PID or VRAM alone is not completion.

- [ ] **Step 3: Validate each intermediate export**

Require three non-empty Safetensors shards, config files, 19 MTP tensors, 124468 total tensor keys, 30971 `*.input_scale` keys (30841 more than W4A16), 331 enabled input quantizers, finite tensors, and different Balanced/Max weight hashes.

---

### Task 6: Merge, upload, and verify one-file W4A4 artifacts

**Files:**
- Create: `balanced/safetensors/Qwen3.6-35B-A3B-Heretic-MOE-Balanced-NVFP4_W4A4.safetensors`
- Create: `max/safetensors/Qwen3.6-35B-A3B-Heretic-MOE-Max-NVFP4_W4A4.safetensors`
- Create: two loader-config folders and two manifests under HF `research/quantization/`

**Interfaces:**
- Consumes: Task 5 three-shard exports.
- Produces: two verified single-file W4A4 LFS objects and loader metadata.

- [ ] **Step 1: Merge shards without changing tensor keys**

Load one shard at a time, reject duplicate keys, save a temporary one-file artifact with metadata `quantization=NVFP4_W4A4_MSE`, reopen it, and assert the merged key set equals the shard union.

- [ ] **Step 2: Upload each file resumably**

Upload to the exact Task 6 paths. Upload config/quantization metadata and manifests separately.

- [ ] **Step 3: Verify remote LFS state**

Compare local and remote byte size/SHA-256, require four explicit NVFP4 paths total, and require no ambiguous old path or sharded weight path.

---

### Task 7: Final card, research evidence, release audit, and shutdown

**Files:**
- Modify: `F:/AI/tmp/qwen36_hf_card/README.md`
- Upload: release inventory, recipe/preflight/conversion manifests, IQ2 exception evidence, journal hashes, and imatrix hashes.

**Interfaces:**
- Consumes: all verified artifact metadata.
- Produces: a self-consistent public release and zero paid compute left running.

- [ ] **Step 1: Update the English model card**

Describe W4A16 as quality-first weight-NVFP4 plus FP8 attention/KV and W4A4 as activation-quantized NVIDIA-scheme NVFP4. Document the IQ2 MTP Q4_K exception and link the Heretic-MOE GitHub repository.

- [ ] **Step 2: Generate the final inventory from HF metadata**

Record every required path, bytes, LFS SHA-256, source variant, format, quantization policy, and validation status. Fail on missing or unexpected shard paths.

- [ ] **Step 3: Run the completion audit**

Verify both variants have BF16, INT8 Lean ConvRot, NVFP4_W4A16, NVFP4_W4A4, F16/Q8/Q6/Q4/IQ4/IQ3/IQ2 GGUF, imatrix, shared mmproj, journals, manifests, and card entries.

- [ ] **Step 4: Destroy paid instance 47358517**

Destroy only after the HF audit passes. Query Vast once more and require the instance to be absent/stopped.
