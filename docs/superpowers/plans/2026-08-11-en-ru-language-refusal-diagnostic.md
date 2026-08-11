# EN/RU Language-Refusal Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible BF16 paired EN/RU behavior and residual-geometry diagnostic before changing the multilingual Heretic-MOE search contract.

**Architecture:** Generate aligned response archives from the original Qwen3.5-9B BF16 model, compare outputs by canonical ID, then run residual analysis only after the behavioral archive passes integrity checks. Keep prompt text local and emit text-free metrics and hashes.

**Tech Stack:** Heretic-MOE, Transformers/PyTorch, JSONL, SQLite, NumPy/SciPy.

## Global Constraints

- Use original Gemma-4 E4B BF16 as the primary model, not a Heretic export or
  GGUF substitute. Use original Qwen3.5-9B BF16 only as the confirmation model.
- Use GPU 1 unless its live allocation changes before launch.
- Greedy decoding, thinking disabled, 256-token cap plus one 512-token retry for capped rows.
- Never print prompt or response text to console or Agent Chat.

---

### Task 1: Freeze and verify the paired corpus

- [ ] Validate EN/RU row counts, unique row IDs, identical ordered canonical IDs,
  matching direction/category metadata, nonempty prompts, and SHA-256 hashes.
- [ ] Write a text-free input manifest; refuse to launch an incomplete direction.
- [ ] If RU SAFE remains incomplete, materialize only the complete EN/RU UNSAFE
  pair set and record SAFE as pending rather than fabricating or mixing rows.

### Task 2: Primary BF16 generation archive

- [ ] Smoke-load Gemma-4 E4B on GPU 1 and verify the real model revision, chat
  template, thinking-off behavior, available VRAM, and one completed EN/RU pair.
- [ ] Generate complete aligned archives at 256 tokens with deterministic settings.
- [ ] Detect length-capped rows mechanically and rerun only those rows at 512.
- [ ] Verify exact key coverage and save model/config/tokenizer/output hashes.

### Task 3: Paired behavioral analysis

- [ ] Calculate text-free completion, length, cap, empty-output, lexical-refusal,
  and available SRG radar features for every row.
- [ ] Join only by canonical ID and produce SAFE/UNSAFE/category transition tables.
- [ ] Build a blind audit packet containing every discordant pair plus a seeded,
  category-stratified concordant control sample.
- [ ] Report paired transition rates and canonical-ID bootstrap intervals; do not
  treat automatic radar labels as ground truth.

### Task 4: Cross-model confirmation

- [ ] Repeat every primary-model discordance and a seeded category-stratified
  concordant control sample on original Qwen3.5-9B BF16.
- [ ] Report Gemma/Qwen agreement on the direction of each EN/RU transition;
  model-specific discordance cannot justify changing the search corpus.

### Task 5: Residual-direction confirmation

- [ ] Reuse the same BF16 checkpoint and aligned IDs to compute EN and RU SAFE and
  UNSAFE residual means by layer.
- [ ] Report per-layer direction cosine, norm ratio, and projection-sign agreement.
- [ ] Cross-check whether layers with the largest language divergence also explain
  audited behavioral discordance.

### Task 6: Decision gate

- [ ] Keep Heretic-MOE English-primary if differences are small, inconsistent, or
  explained by translation errors.
- [ ] Add RU only as a low-weight diagnostic if behavior differs without stable
  residual support.
- [ ] Allow RU into the optimization objective only when both behavioral and
  residual evidence agree and SAFE over-refusal does not worsen.
