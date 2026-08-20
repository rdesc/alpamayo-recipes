# Baseline eval: 19-chunk nav subset (stride 2.0s)

Run of `scripts_fork/eval_baseline_pai19chunks.py` (no-RL baseline) against the
full 19-chunk nav subset (`clip_index_19chunks_valid.parquet`), sliding-window
t0 with `--t0-stride-s 2.0`, single trajectory sample per window.

> **2026-08-19/20: superseded prompt fix.** The numbers below use the corrected prompt
> construction (`alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config`,
> matching actual RL training's `components_order`/`chat_template_version`, including
> camera-identity/frame-index prompt text). The original run (preserved at the bottom) used
> `alpamayo_r1.helper.create_message`, which omits that annotation entirely. See
> `scripts_fork/results/eval_prompt_format_discrepancy.md` for the investigation.

- Model: `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training`
- CoC extractor: `Qwen/Qwen3-VL-8B-Instruct`
- `--num-traj-samples 1`, `--t0-stride-s 2.0`
- Sharded 4-way across GPUs 4-7, ~553-591 min wall clock per shard (via `a1x_rl_b300` venv --
  this recipe's default `a1x_rl` venv fails on every event on this B300 box with
  `nvrtc: invalid value for --gpu-architecture`; unrelated to this script, see
  `eval_prompt_format_discrepancy.md`)
- Output: `/opt/dlami/nvme/rod/results/baseline_pai19chunks_stride2s_v2/baseline.shard{0..3}-of-4.parquet`

## Per-shard run summary

| shard | GPU | wall time | rows ok | ADE mean | coc_consistency | abstain rate |
|---|---|---|---|---|---|---|
| 0 | 4 | 583.9 min | 4651/4651 | 1.717 m | 0.747 | 0.000 |
| 1 | 5 | 590.9 min | 4652/4652 | 1.755 m | 0.738 | 0.001 |
| 2 | 6 | 553.4 min | 4604/4604 | 1.657 m | 0.745 | 0.000 |
| 3 | 7 | 576.5 min | 4652/4652 | 1.651 m | 0.734 | 0.000 |

No errors in any shard log. 18,559 total windows / 1,877 unique clips, 100% ok.

## Overall (n=18,559)

| metric | mean | median |
|---|---|---|
| ADE / min-ADE (K=1) | 1.70 m | 1.11 m |
| min-ADE @ t=0.5s | 0.014 m | 0.009 m |
| min-ADE @ t=1.0s | 0.045 m | 0.028 m |
| min-ADE @ t=3.0s | 0.372 m | 0.246 m |
| min-ADE @ t=6.0s | 1.491 m | 0.976 m |

CoC-action-consistency (model's own predicted CoC vs. its own predicted trajectory):

- coc_binary accuracy: **0.732**
- coc_graded (soft score): **0.741**
- coc_reward_contribution: **0.741**
- abstain rate: ~0.04%

## By split

| split | n | ADE mean | coc_binary mean |
|---|---|---|---|
| train | 8,816 | 1.73 m | 0.761 |
| val | 6,808 | 1.71 m | 0.697 |
| test | 2,935 | 1.55 m | 0.721 |

## By chunk (19 chunks, ~942-996 windows each)

| chunk | n | ADE mean | coc_binary mean |
|---|---|---|---|
| 214 | 986 | 1.72 m | 0.656 |
| 224 | 995 | 1.76 m | 0.689 |
| 276 | 942 | 1.75 m | 0.566 |
| 317 | 972 | 1.74 m | 0.651 |
| 420 | 990 | 1.99 m | 0.734 |
| 727 | 983 | 1.91 m | 0.747 |
| 728 | 963 | 1.88 m | 0.731 |
| 968 | 994 | 1.80 m | 0.719 |
| 982 | 995 | 2.11 m | 0.723 |
| 1519 | 971 | 1.05 m | 0.841 |
| 1657 | 955 | 1.26 m | 0.814 |
| 1984 | 952 | 1.15 m | 0.851 |
| 2277 | 977 | 1.15 m | 0.822 |
| 2368 | 996 | 1.88 m | 0.724 |
| 2372 | 993 | 1.73 m | 0.720 |
| 2447 | 989 | 2.03 m | 0.779 |
| 2599 | 944 | 1.65 m | 0.676 |
| 2634 | 986 | 1.78 m | 0.690 |
| 2868 | 976 | 1.82 m | 0.765 |

Same pattern as before the fix: chunk 276 remains weakest on CoC accuracy (0.566);
chunks 1519/1657/1984/2277 remain the standout cluster with both lower ADE (~1.1-1.3m) and
higher CoC accuracy (0.82-0.85). ADE dropped ~26% overall (2.29m -> 1.70m mean); CoC-consistency
is essentially unchanged (0.731 -> 0.732 binary), as expected.

---

## Superseded: original run (pre prompt-fix, kept for reference)

Numbers below used `alpamayo_r1.helper.create_message`, which omits camera-identity/frame-index
prompt annotation. See the note at the top of this doc.

- Sharded 4-way across GPUs 4-7, ~509-535 min wall clock per shard
- Output: `/opt/dlami/nvme/rod/results/baseline_pai19chunks_stride2s/baseline.shard{0..3}-of-4.parquet`

| shard | GPU | wall time | rows ok | ADE mean | coc_consistency | abstain rate |
|---|---|---|---|---|---|---|
| 0 | 4 | 534.7 min | 4651/4651 | 2.320 m | 0.744 | 0.000 |
| 1 | 5 | 485.8 min | 4652/4652 | 2.341 m | 0.736 | 0.000 |
| 2 | 6 | 511.6 min | 4604/4604 | 2.252 m | 0.726 | 0.001 |
| 3 | 7 | 509.1 min | 4652/4652 | 2.251 m | 0.742 | 0.001 |

| metric | mean | median |
|---|---|---|
| ADE / min-ADE (K=1) | 2.29 m | 1.58 m |
| min-ADE @ t=0.5s | 0.019 m | 0.014 m |
| min-ADE @ t=1.0s | 0.061 m | 0.043 m |
| min-ADE @ t=3.0s | 0.521 m | 0.364 m |
| min-ADE @ t=5.0s | 1.422 m | 0.983 m |
| corner distance | 2.35 m | 1.65 m |

- coc_binary accuracy: 0.731
- coc_graded: 0.738
- coc_reward_contribution: 0.737
- abstain rate: ~0.05%

| split | n | ADE mean | coc_binary mean |
|---|---|---|---|
| train | 8,816 | 2.354 m | 0.758 |
| val | 6,808 | 2.289 m | 0.700 |
| test | 2,935 | 2.111 m | 0.725 |

| chunk | n | ADE mean | coc_binary mean |
|---|---|---|---|
| 214 | 986 | 2.329 m | 0.677 |
| 224 | 995 | 2.412 m | 0.726 |
| 276 | 942 | 2.423 m | 0.563 |
| 317 | 972 | 2.418 m | 0.664 |
| 420 | 990 | 2.717 m | 0.733 |
| 727 | 983 | 2.541 m | 0.774 |
| 728 | 963 | 2.575 m | 0.734 |
| 968 | 994 | 2.463 m | 0.712 |
| 982 | 995 | 2.613 m | 0.712 |
| 1519 | 971 | 1.576 m | 0.828 |
| 1657 | 955 | 1.652 m | 0.803 |
| 1984 | 952 | 1.636 m | 0.810 |
| 2277 | 977 | 1.626 m | 0.835 |
| 2368 | 996 | 2.672 m | 0.706 |
| 2372 | 993 | 2.410 m | 0.715 |
| 2447 | 989 | 2.545 m | 0.779 |
| 2599 | 944 | 2.115 m | 0.695 |
| 2634 | 986 | 2.289 m | 0.675 |
| 2868 | 976 | 2.462 m | 0.754 |
