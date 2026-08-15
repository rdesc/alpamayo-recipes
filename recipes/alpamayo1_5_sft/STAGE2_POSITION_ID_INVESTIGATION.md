# Stage-2 expert: is the batched `<traj_future_start>` crop a bug?

> **Update:** a real, confirmed batching bug was found and reproduced — see
> [Confirmed bug: left-padding leaks into the diffusion expert's rollout
> attention](#confirmed-bug-left-padding-leaks-into-the-diffusion-experts-rollout-attention)
> below. It lives in a *different* code path than the one this doc originally
> investigated (rollout/inference, not the batched training `forward()`), and
> unlike the training-side finding, this one is live today, not just fragile.
> The original investigation (training's KV-cache crop) is kept below for the
> record; its verdict (not currently a bug) stands.

Investigation prompted by a claim (from a separate chat, while comparing
eval's per-sample VLM rollout against Stage-2 training's batched
teacher-forced pass) that `TrainableAlpamayoR1.forward()` computes one
`<traj_future_start>` position from an arbitrary batch row and applies it to
every row, and that this is "plausible a live, unnoticed bug" masked by SGD
tolerating noisy gradients. Verdict below: **not currently a bug, but an
unenforced invariant that would silently reintroduce this exact failure mode
if broken.**

## The code in question

`recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:167-189`
(`TrainableAlpamayoR1.forward`):

```python
future_start_token_id = self.config.traj_token_ids["future_start"]
last_traj_future_start_idx = (input_ids == future_start_token_id).nonzero(as_tuple=False)
last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1
...
kv_cache = vlm_outputs.past_key_values
kv_cache.crop(last_traj_future_start_idx)          # ONE scalar for the whole batch
...
position_ids = self._process_position_ids_qwen2_5_vl(
    vlm_outputs, batch_size, expert_embeds.shape[1], expert_embeds.device
)
```

`_process_position_ids_qwen2_5_vl` (`sft_alpamayo_r1.py:106-124`):

```python
position_ids = torch.arange(num_expert_tokens, device=device)
position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
position_ids += delta.to(position_ids.device)
```

`torch.nonzero` on a 2D tensor walks batch-major, position-minor, so
`[-1, 1]` is the `<traj_future_start>` position from whichever sample
happens to be **last in the batch dimension** — not a max, not a per-sample
value. That scalar then:

1. crops the (single, batched) KV-cache tensor for every row, and
2. is added, via `kv_cache.get_seq_length()`, to every row's `position_ids`
   (on top of `rope_deltas`, which *is* per-sample).

If different rows had `<traj_future_start>` at different absolute indices,
this would crop away real context for some rows and leave stale/padding
context for others, and would offset `position_ids` incorrectly for all
rows but the one the index was taken from.

## Why it's not actually broken today

Two things have to both hold for the "last row's index applies to every
row" scalar to be correct, and both hold in every config currently in this
fork:

**1. Left padding right-aligns each sample's real content.**

`collate_fn` (`qwen_processor.py:168-184`, reached via
`collate_fn_from_model_config`, `configs/sft_base.yaml:7-9`) pads with
`padding_side="left"` by default, and nothing in this repo overrides it.
Left padding means `[PAD]*k + real_tokens`, so every sample's real content
ends at the same absolute index (`max_len - 1`) regardless of how long its
(variable-length) reasoning text is.

**2. `traj_future` is the last component, so every sample has an identical
fixed-length tail after `<traj_future_start>`.**

`build_conversation` / `construct_traj_future`
(`.../alpamayo_r1/chat_template/conversation.py:157-179, 271-273`) only
emits the bare `<traj_future_start>` token (no padding placeholders, no end
tag) when the component is being "asked for" — i.e.
`generation_mode=True` *and* it's the last entry of `components_order`.
Otherwise it emits `<traj_future_start>` + `tokens_per_future_traj`
placeholder tokens (a fixed model-config constant, 128 by default) +
`<traj_future_end>`.

Checked every `components_order` that reaches this batched `forward()`
(i.e. every teacher-forced train/eval-loss pass, as opposed to the
free-running rollout eval path):

| Config | `components_order` | `traj_future` last? |
|---|---|---|
| `configs/vla_processor/default.yaml` | `image, traj_history, prompt, traj_future` | yes |
| `configs/vla_processor/nav.yaml` (used by `sft_stage2_navsim.yaml`) | `image, traj_history, route, prompt, traj_future` | yes |
| `sft_stage2_truckdrive.yaml` (inherits `sft_truckdrive.yaml`) | same as `default.yaml` | yes |

`sft_stage2_truckdrive_cot.yaml` overrides `components_order` to end at
`cot`, but that config only drives the free-running
`sample_trajectories_from_data_with_vlm_rollout` eval path (per-sample
generation, not this batched `forward()`), so it doesn't exercise the code
above at all — see its own header comment.

**Given both invariants**, every row's real content is right-flush to the
same absolute end index, and every row has the same fixed-length suffix
after `<traj_future_start>` (whether that suffix is empty, at eval with
`generation_mode=True`, or the constant placeholder+end-tag block, at
train). So `<traj_future_start>`'s absolute index is **identical across
every row in the batch** — reading it off the last row and applying it to
all rows is equivalent to computing it per-sample, not a shortcut that
happens to get lucky.

This is also arguably the only design that *could* work here: HF's KV
cache is one shared `[batch, heads, seq, head_dim]` tensor, so `crop()`
can't take a per-sample crop point regardless. The padding/component-order
scheme is what makes a single shared crop point valid, not an oversight.

## Where this is fragile

None of the above is enforced in code — it's a convention that happens to
hold across every config in the repo today. It would break silently (same
failure mode described in the original bug report: wrong KV-cache context
and wrong `position_ids` for every row but the last, a noisy-but-not-crashing
gradient corruption) if:

- `padding_side` is ever changed to `"right"` (e.g. someone "standardizes"
  padding without knowing this file depends on left-padding), or
- a training/loss config's `components_order` ever puts something after
  `traj_future`, or ends at `traj_future` while *not* last (e.g. a future
  Stage-2 CoC-training variant, as opposed to today's CoC-eval-only
  `sft_stage2_truckdrive_cot.yaml`), or
- `tokens_per_future_traj` ever became per-sample instead of a fixed model
  config scalar.

No assertion currently guards this, and there's no test covering it
(`recipes/alpamayo1_5_sft/tests/` has nothing for `sft_alpamayo_r1.py`).

## Fix applied

Added a defensive assert in `forward()` (`sft_alpamayo_r1.py:167-186`)
right where `last_traj_future_start_idx` is computed: it now checks there's
exactly one `<traj_future_start>` per sample and that its index matches
across every row in the batch before using the last row's value. If the
invariants above ever break (padding side, `components_order`, or a
per-sample-variable `tokens_per_future_traj`), training now fails loudly at
the first affected step instead of silently corrupting gradients for part
of every batch.

---

## Confirmed bug: left-padding leaks into the diffusion expert's rollout attention

Follow-up investigation, triggered by an experiment showing Stage-2 rollout
predictions differ between batched and single-sample inference. Traced to a
**different** method than the one above — `forward()` computes training
*loss*; batched *inference* goes through
`AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout`
(`alpamayo_r1/models/alpamayo_r1.py:124-250`, vendored into this fork's
Stage-2 SFT venv at
`recipes/alpamayo1_5_sft/a1_5_sft/lib/python3.12/site-packages/alpamayo_r1/models/alpamayo_r1.py`).
This one **is** a live, confirmed bug.

### The gap

After the VLM free-runs its CoC generation, the diffusion expert
cross-attends to the frozen VLM's KV-cache to condition its denoising steps.
Since different rows in a batch generate `<traj_future_start>` at different
steps, the code (correctly) can't use one shared KV-cache crop point (unlike
training's teacher-forced case above) — instead it builds a per-row 4D
attention mask (`alpamayo_r1.py:241-250`):

```python
attention_mask = torch.zeros(
    (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
    dtype=torch.float32, device=device,
)
for i in range(b_star):
    attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
        attention_mask.dtype
    ).min
```

This masks the **trailing** region after each row's own `<traj_future_start>`
(the "over-generated" tail from rows that kept decoding past their own stop
point because `StopAfterEOS` only halts the whole batch once *every* row has
found the token). It never touches columns `[0 : offset[i])`. Since
`collate_fn` left-pads (`padding_side="left"`, the default, never overridden
in this fork), that untouched region includes every row's left-padding
whenever it's shorter than the batch's longest prompt. The expert ends up
attending to pad-token KV states as if they were real context.

At `batch_size=1` there's nothing to pad against, so this is invisible —
which is exactly why it can slip past testing that only ever exercises
single-sample generation.

### Experimental confirmation

Two experiments, run against the trained `output_stage2_navsim_seed0`
checkpoint (`/home/rod/ckpts/alpamayo_navsim_finetuning/output_stage2_navsim_seed0`)
on real navtrain val windows, with `FlowMatching._euler`'s initial noise
monkeypatched to a pinned value so the only difference between compared runs
is padding, not RNG:

**1. Batched (B=4) vs single (B=1), same scenes, same pinned noise per
scene:**

| scene | pad tokens | role | max\|Δ\| | endpoint Δ |
|---|---|---|---|---|
| 23 | 0 | control | 0.019 m | 0.013 m |
| 21 | 0 | control | 0.046 m | 0.048 m |
| 3 | 4 | test | 0.069 m | 0.055 m |
| 4 | 4 | test | 0.135 m | 0.110 m |

Padded rows moved ~2.9x more than the zero-padding control — suggestive, but
the control itself already drifts up to 4.6 cm from ordinary bf16
batch-size-dependent kernel reduction noise, so this alone doesn't isolate
the cause.

**2. Isolation: same single scene, `B=1` in every run, same pinned noise —
only the amount of artificial left-padding prepended to `input_ids`/
`attention_mask` differs:**

| left-pad tokens | max\|Δ\| | mean\|Δ\| | endpoint Δ |
|---|---|---|---|
| 0 (reference) | — | — | — |
| 4 | 0.045 m | 0.006 m | 0.032 m |
| 32 | 0.253 m | 0.035 m | 0.208 m |
| 256 | 1.390 m | 0.207 m | 1.074 m |

With batch size held at 1 throughout, RNG divergence and bf16 batch-kernel
effects are both ruled out by construction — the only variable is padding
width, and deviation scales with it, reaching **1.39 m** of max trajectory
error at 256 pad tokens (a realistic gap between a short and long CoC
reasoning trace in the same batch). This is not noise; it's the padding leak
predicted from reading the code, reproduced and quantified.

Repro scripts:
`/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/4c9c87b6-8c7a-4a1b-a889-5cfa2d190291/scratchpad/{batch_vs_single,padding_isolation}.py`
(scratch dir — not checked in).

### Blast radius in this fork

Every batched eval/viz call site goes through this same rollout method with
`batch_size>1` by default:

| Call site | Default batch size |
|---|---|
| `callbacks/val_metric_callback.py:58` (`val/minADE_*` metrics) | 8 |
| `callbacks/viz_callback.py:107` (W&B trajectory viz) | 4 |
| `truckdrive/score_val_ade.py:73` | 8 |

Any batch mixing scenes with meaningfully different prompt lengths (variable
CoC/reasoning length, variable nav-route text, variable image/camera counts)
will have some rows' predicted trajectories corrupted by this leak — meaning
reported val minADE / ADE numbers from this fork's Stage-2 runs are
systematically noisier (and directionally worse) than they should be,
proportional to how much length variance exists in each batch.

### It's already fixed upstream — just not in this fork's vendored copy

`~/repos/alpamayo1.5` (the canonical NVIDIA release repo, a newer generation
of the same code) already handles this correctly, and has since its first
commit (`3eafdd3`, "Release Alpamayo-1.5-10B model") —
`src/alpamayo1_5/models/alpamayo1_5.py`'s `_build_expert_pos_ids_and_attn_mask`
takes an explicit `prefix_mask` and additionally masks `[0:L)` wherever the
original tokenizer attention mask is 0:

```python
if prefix_mask is not None:
    input_mask = prefix_mask[:, None, None, :]
    attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
        input_mask == 0, torch.finfo(attention_mask.dtype).min,
        attention_mask[:, :, :, : input_mask.shape[-1]],
    )
```

called with `prefix_mask = torch.repeat_interleave(tokenized_data["attention_mask"], n_samples_total, dim=0)`.
This fork's Stage-2 SFT venv simply vendors an older/different `alpamayo_r1`
package that predates this fix.

### Fix applied

Added `TrainableAlpamayoR1.sample_trajectories_from_data_with_vlm_rollout`
(`sft_alpamayo_r1.py`) — a fork-local override of the vendored method, adding
the same `prefix_mask` step `alpamayo1.5` already has (option 2 above:
versioned in git, survives venv reinstalls, mirrors how `forward()` is
already a local override of the base class).

**Verified the fix with the same isolation experiment** (same scene, `B=1`,
same pinned noise, only left-padding width varies), before vs. after:

| left-pad tokens | max\|Δ\| before | max\|Δ\| after |
|---|---|---|
| 4 | 0.045 m | 0.023 m |
| 32 | 0.253 m | 0.016 m |
| 256 | 1.390 m | 0.030 m |

The signature that mattered is gone: before the fix, error grew
monotonically with padding width (each 8x increase in padding gave ~5-6x
more error). After the fix, error is flat/non-monotonic across padding
widths (0.023 → 0.016 → 0.030 m) — no scaling with pad width — and sits in
the same ~0.02-0.05 m band as the bf16 batch-kernel noise floor established
by the zero-padding control rows in the batch-vs-single experiment above.
What's left is ordinary numerical noise, not the leak.

### One more latent issue noticed in passing (not confirmed to be hit)

`step_fn` (both this fork's vendored copy and `alpamayo1.5`'s version) closes
over `position_ids`/`attention_mask`/`prompt_cache` sized for
`b_star = B * num_traj_samples` (the VLM generation's own
`num_return_sequences`), but `self.diffusion.sample(batch_size=B *
num_traj_samples * num_traj_sets, step_fn=step_fn, ...)` drives `step_fn`
with that larger size whenever `num_traj_sets > 1`. Every call site checked
in both repos (`val_metric_callback.py`, `viz_callback.py`,
`score_val_ade.py`, `sft_base.yaml`, and everything in `alpamayo1_5`'s own
tree) hardcodes `num_traj_sets=1`, so this is currently dormant, not
confirmed to cause a real failure — flagged here so it isn't rediscovered
from scratch if `num_traj_sets>1` is ever used.
