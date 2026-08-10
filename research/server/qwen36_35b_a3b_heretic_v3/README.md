# Qwen3.6-35B-A3B Heretic-MOE v3 search bundle

This directory contains the complete search-only server bundle. It downloads
Heretic-MOE from GitHub and the frozen base model from Hugging Face, verifies
the two-GPU runtime and private data sidecar, runs 120 alternating Random/Sobol
trials, continues the shared multivariate-TPE study to 600 completed trials,
and rechecks the top six at 64 x 1024. It does not export or quantize models.

## 1. Clone Heretic-MOE on the server

```bash
git clone https://github.com/dborzoff/Heretic-MOE.git /workspace/Heretic-MOE
cd /workspace/Heretic-MOE/research/server/qwen36_35b_a3b_heretic_v3
```

The controller records the exact checked-out Git revision in its run manifest.

## 2. Upload the private scorer data from Windows

Run locally from this directory:

```powershell
.\upload_data.ps1 -ValidateOnly
.\upload_data.ps1 -HostName SERVER -Port SSH_PORT -IdentityFile KEY
```

The first command performs the same frozen size and SHA-256 checks without
opening an SSH connection. The upload command repeats those checks before
transfer. Prompt and response contents are not printed or committed.

## 3. Prepare and run in one visible server console

```bash
read -rsp "HF token: " HF_TOKEN && echo
export HF_TOKEN
./prepare_and_run.sh
```

The environment installation and the pinned Hugging Face download run in
parallel with `[ENV]` and `[MODEL]` prefixes. Search begins only after package,
model-shard, data-hash, CUDA-count, and VRAM checks pass.

Outputs are written to `/workspace/heretic-runs/qwen36-35b-a3b-v3`. The final
step creates a high-fidelity top-six report but does not assemble model weights.

## Resume or extend

Running the same command resumes the immutable journal. To extend an existing
600-trial run after reviewing the top six:

```bash
TARGET_TRIALS=1000 ./run_search.sh
```

The dynamic worker queue keeps one resident model per GPU and allocates the next
trial to whichever worker becomes available first.
