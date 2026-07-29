# What this fork changes

Heretic works by removing a refusal direction from a model's output
projections. This fork makes that work on fused-MoE architectures, and changes
what the search optimises for.

Changes live on `master`, so `git clone` is all a machine needs. Upstream is
tracked as `upstream` and rebases cleanly: nothing in the original README or
CLI surface is touched.

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
expensive part.

The split is done by **adding** two keys rather than renaming the existing ones:

| key | what lands there | which models see it |
|---|---|---|
| `attn.o_proj` | full attention output | all, unchanged |
| `attn.linear.out_proj` | linear attention output | only hybrids that have one |
| `mlp.down_proj` | dense MLP / routed experts | all, unchanged |
| `mlp.shared.down_proj` | shared expert | only models that have one |

Any architecture without a linear-attention or shared-expert path therefore
sees exactly the two keys it saw before, with the same names — which matters
because parameter names are what a stored study keys on, so renaming them would
break `--checkpoint-action continue` on existing studies. The lower-bound rule
keys off component meaning (`mlp.` prefix or `linear` in the name) rather than
an exact string, for the same reason.

Search bounds are widened at the same time, and this is **the one change that
affects every model**, not just MoE ones. The old bounds clipped the best
trials: winners pressed against three of four limits. After widening, the
record moved from **0.15 to 0.01 refusals**, with the edit peaking at layer 15
of 40 — far earlier than the old floor of 0.6-of-depth allowed.

```
max_weight            1.5              ->  2.5
max_weight_position   0.6..1.0 depth   ->  0.0..1.0 depth
min_weight_distance   1.0..0.6 depth   ->  1.0..1.5 depth
```

Nothing is taken away: the old optimum is still inside the new space. But the
space is larger, so on a model where the old bounds happened to be right, a
fixed trial budget will spend some of itself confirming that. Worth knowing
before running this on something other than a hybrid MoE.

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

## 4. A warning when the front presses against the bounds

Bounds encode an assumption about where the answer can lie. When the assumption
is wrong, the search quietly hits a wall and returns the best of what it was
*allowed* — which looks exactly like the best there is, if you only read the
numbers. You have to look at where the winning points ended up.

That cost us a whole search before we noticed. This checks the front after the
run and says so:

```
Лучшие точки жмутся к границам поиска:
  * mlp.down_proj.max_weight: 7 из 9 точек фронта у верхнему пределу (1.500)
```

Moving bounds mid-study is *not* the fix and is not attempted here: a
distribution is fixed when the study is created and TPE builds its model on it,
so changing it invalidates everything the sampler learned about that parameter.
The check only reports, and points at the cheap way to act on it — restart with
wider bounds, seeded from the points already found.

## 5. Seeding a new study from a previous one

`--seed-trials-from <journal>` enqueues the front points of an earlier study
into a new one. This is what makes a changed objective affordable: stored
objective values become meaningless and the study has to start over, but the
parameter sets that reached the front are still the best guess available.

Two things it has to get right, both of which fail silently otherwise — Optuna
rejects the whole enqueued set and the run dies before its first trial. A
journal stores the *internal* representation of a parameter, so a categorical
is an index and has to be mapped back to its choice. And parameters whose
component no longer exists must be dropped, with one rename applied
deliberately: routed experts used to share a key with the shared expert and now
have their own, and by mass that key meant the routed experts anyway.

Default is twelve points, and deliberately modest. A seed helps most when the
search space is unchanged; ours is not, so a seeded point fixes only the
parameters that still mean what they used to and the rest get sampled anyway. A
large seed spends the budget re-checking half-random points instead of exploring
what is actually new.

## 6. CLI settings survive `--checkpoint-action continue`

Settings were restored wholesale from the study's stored JSON, discarding what
was passed on the command line. The failure was quiet and expensive:
`save_directory` is excluded from stored settings, so a resumed run prompted
for it interactively and died unattended after hours of search.

## Compatibility

Component detection was run against seventeen local models, using this fork's
own class selection and layer lookup. Fifteen see exactly the two keys upstream
gives them — `attn.o_proj` and `mlp.down_proj` — so nothing about their
behaviour changes:

> Ministral-3-3B (both), Qwen2.5-3B-Instruct, Qwen2.5-VL-7B-Instruct,
> Qwen3-0.6B-Base, Qwen3-4B, Qwen3-8B, Qwen3-VL-4B, Qwen3-VL-8B,
> ERNIE-Image-pe, gemma-2-2b, gemma-2-2b-it, gemma-3-12b-it, gemma-4-E4B-it

Two are hybrids and pick up the split: Qwen3.5-9B gains
`attn.linear.out_proj`, and Qwen3.6-35B-A3B gains that plus
`mlp.shared.down_proj`, with its fused experts on `mlp.experts.down_proj`.

The full pipeline — direction, ablation, both scorers, rollback between trials,
front selection — has been run end to end on gemma-2-2b-it, a plain dense
model, as a check that none of this disturbs the ordinary path.

**The widened bounds are the one change that affects every model.** Nothing is
taken away, but the space is larger, so a fixed trial budget explores it more
thinly. Worth knowing before running this on something that was already well
served by the original bounds.

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
