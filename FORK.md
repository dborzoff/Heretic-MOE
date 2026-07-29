# What this fork changes

Heretic works by removing a refusal direction from a model's output
projections. This fork makes that work on fused-MoE architectures, and changes
what the search optimises for.

Branch: `moe`. Upstream is tracked as `upstream`; the branch rebases cleanly
because nothing in the original README or CLI surface is touched.

## 1. Fused MoE experts

Qwen3.5/3.6 MoE stores its 256 routed experts as `nn.Parameter` tensors of
shape `[num_experts, out_features, in_features]`, not as a `ModuleList` of
`Linear` layers. Heretic walks a model looking for linear modules, so on these
architectures the routed experts — **92% of the weights** — are passed over in
silence and only attention is edited.

The experts are where the model lives. In our 528-trial search on
Qwen3.6-35B-A3B, trials that left the MLP weight near zero (attention-only, the
way abliteration works without this patch) never got below **59 refusals out of
100**, across 57 such trials. Everything under 12 needed the experts.

Applied per expert in chunks of 32 to bound peak memory; originals are cached
so a trial can be undone.

## 2. One weight schedule per component that deserves one

Heretic builds one set of four numbers — peak weight, its position, the minimum
and the span — per *component key*. On this architecture each key covers two
structurally different things:

| key | what is actually under it |
|---|---|
| `attn.o_proj` | `self_attn.o_proj` (10 layers, 0.27B) + `linear_attn.out_proj` (30 layers, 1.01B) |
| `mlp.down_proj` | routed experts, 8 of 256 fire per token, 32.21B + shared expert, fires always, 0.13B |

The model is a hybrid: only every fourth layer (3, 7, 11 … 39) carries full
attention; the other thirty use linear attention. Linear attention holds four
times the weight and three times the layers, yet moves on the same knob. The
shared expert is 250× lighter than the routed ones but runs on every token, so
what its edit costs per unit of weight is not comparable.

With one knob the search cannot keep the useful part of an edit and drop the
expensive part. This splits the keys into four and fixes the lower-bound rule
to key off component *meaning* rather than an exact string, so other
architectures keep their previous behaviour exactly.

Search bounds are widened at the same time. The old ones clipped the best
trials — winners pressed against three of four limits. After widening, the
record moved from **0.15 to 0.01 refusals**, with the edit peaking at layer 15
of 40, far earlier than the old floor of 0.6-of-depth allowed.

## 3. Perplexity as an objective

`KLDivergence` takes 100 harmless prompts and compares the distribution of the
**first token** of the reply. One hundred numbers. It sees whether the opening
of an answer changed, and is nearly blind to what follows.

Measured on the two builds we published from this search:

| build | KL (first token) | perplexity vs base | refusals |
|---|---|---|---|
| balanced | 0.0021 | **+3.4%** | 5–8/100 |
| max | 0.0126 | **+18.1%** | 2–4/100 |

KL puts them a factor of six apart; the text says the aggressive build costs
five times more in a unit that describes what the model actually lost. The
search was ranking points on a curve whose real cost it could not see.

Perplexity was not usable as an objective before because it was too slow — a
full pass took fourteen minutes on CPU, which does not fit in a five-hundred
trial loop. On a GPU it takes seconds. `heretic.scorers.perplexity.Perplexity`
measures it on fixed windows of wikitext and returns the relative increase over
baseline.

```toml
[[scorers]]
plugin = "heretic.scorers.perplexity.Perplexity"
optimization = "minimize"
```

## 4. CLI settings survive `--checkpoint-action continue`

Settings were restored wholesale from the study's stored JSON, discarding what
was passed on the command line. The failure was quiet and expensive:
`save_directory` is excluded from stored settings, so a resumed run prompted
for it interactively and died unattended after hours of search.

## Not done yet

Three levers this architecture has that abliteration never touches:

**The router.** `mlp.gate` is 21M parameters, 0.06% of the model, and decides
which 8 of 256 experts fire. Row norms vary by a factor of 1.8–1.9 at every
depth, so layers do call some experts far more readily than others. If refusal
is carried by particular experts, steering the router is an edit to 21M weights
instead of 32B. Untested, and the largest remaining opportunity.

**The gates.** `shared_expert_gate` is `[1, 2048]` — one scalar per token
controlling an always-active component. Full attention takes its gate from half
of `q_proj` (8192 wide against a 4096 output); linear attention from
`in_proj_z`. All three multiply directly into the residual stream, none are
touched.

**The MTP layer.** It carries a full copy — its own 256 experts, router,
attention and shared expert, 0.84B parameters — and is restored verbatim from
the base model, so that branch remains uncensored-unmodified. Under speculative
decoding the main model verifies proposals, so the output distribution should
be the main model's, but this is worth measuring rather than assuming.

## Provenance

The search journal (528 trials) and both published builds:

- [Qwen3.6-35B-A3B-heretic-study](https://huggingface.co/datasets/DmitryDB/Qwen3.6-35B-A3B-heretic-study)
- [Qwen3.6-35B-A3B-heretic-moe-max](https://huggingface.co/DmitryDB/Qwen3.6-35B-A3B-heretic-moe-max)
- [Qwen3.6-35B-A3B-heretic-moe-balanced](https://huggingface.co/DmitryDB/Qwen3.6-35B-A3B-heretic-moe-balanced)
