# Qwen3.6 Heretic-MoE-v3 Hugging Face Card Design

## Purpose

Publish one English model card for the complete `Qwen3.6-35B-A3B Heretic-MoE-v3`
release. The repository may contain distinct Balanced and Max checkpoints or a
single physical checkpoint selected for both roles.

## Card structure

The repository root contains the only model card, `README.md`. It includes:

1. Hugging Face metadata for the upstream model, license, pipeline, and tags.
2. A concise release description and a link to
   <https://github.com/dborzoff/Heretic-MOE>.
3. One release-variant table covering Balanced and Max.
4. The search and high-fidelity recheck contract.
5. An artifact table covering BF16, F16 GGUF, mmproj, model-specific imatrix,
   Q8/Q6/Q4/IQ GGUF files, and only architecture-supported GPU formats.
6. Compatibility limits and reproducibility evidence.
7. Exact final trial identifiers, metrics, file sizes, SHA-256 hashes, and tool
   revisions after those values have been verified.

No additional `README.md` files are created in Balanced, Max, GGUF, or research
subdirectories.

## Variant rules

- Balanced is the preservation-oriented winner among candidates that pass the
  refusal-removal and preservation gates.
- Max is the strongest refusal-removal winner that passes the preservation
  gates.
- If both roles resolve to one physical checkpoint, the card identifies Max as
  an alias of Balanced and the release does not duplicate model weights.
- A role is not described as released until its export manifest reports `PASS`
  and the uploaded files have been independently verified.

## Publication rules

- All prose is English.
- The public card contains no evaluation prompts or generated responses.
- Search metrics, parameters, manifests, counts, hashes, and tool revisions may
  be published.
- While the build is running, the card explicitly says `Build in progress` and
  does not invent final measurements or supported formats.
- Unsupported or unvalidated INT8, NVFP4, W8W4, or text-encoder claims are
  omitted rather than inferred from file size or static conversion success.

## Acceptance checks

The staged card must have valid Hugging Face YAML, one H1 title containing
`Heretic-MoE-v3`, the GitHub source link, exactly one Balanced/Max table, no
private text, and no claim contradicted by the final manifests. After upload,
every published artifact row must reference a verified path, byte size, and
SHA-256 digest.
