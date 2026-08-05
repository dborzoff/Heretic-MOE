# Gemma 3 12B QAT Heretic-MOE / LTX 2.3 run

1. Verify the downloaded five-shard QAT BF16 source and both RTX 5090 devices.
2. Run one isolated trial on GPU 0 and record elapsed time, peak VRAM, sparse refusal geometry, PPL, and LTX all-layer conditioning drift.
3. If the measured cost is acceptable, run 120 exploration trials: 60 Random on GPU 0 and 60 scrambled Sobol on GPU 1.
4. Merge both exploration journals deterministically and continue multivariate TPE on both GPUs to 600 total trials.
5. Preserve the shared journal and response archive. Export several Pareto/feasible finalists; do not select from the marker score alone.
6. Re-evaluate finalists at the full validation budget and reject candidates with excessive PPL or LTX-conditioning drift.
7. Build the selected BF16 master, LTX 2.3 BF16/INT8-ConvRot/NVFP4 variants, and one F16 GGUF plus its model-specific imatrix.
8. Upload and verify hashes before deleting any server artifact. Build Q8/Q6/Q4/IQ4/IQ3/IQ2 GGUF locally from F16 + imatrix.

The prompt/response texts are not inspected by Codex; only files, hashes, counts, and numeric metrics are handled.
