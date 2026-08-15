# B300 migration notes (Cosmos-RL / vLLM 0.20.0)

This documents the full debugging history of getting `alpamayo1_x_rl`'s RL
training running on the new B300 GPUs. Read this before touching the
`a1x_rl_b300` venv or chasing a new crash — it is very likely a known issue
below, not a new one.

## The fundamental issue

These are B300 SXM6 GPUs: **compute capability 10.3 / architecture string
`sm_103a`** (NVIDIA Blackwell Ultra). This is very new hardware, and "does
vLLM support B300" is not a single yes/no fact — it's a chain of
**independently-versioned** packages that all need to support the hardware
*and* stay API-compatible with each other, and no published snapshot has
all of that true at once:

- vLLM depends on `nvidia-cutlass-dsl` as an open-ended floor
  (`>=4.4.2`, no ceiling in vLLM 0.20.0's metadata). vLLM's own vendored
  kernel code was written against cutlass-dsl's API as of roughly 4.5.x.
- cutlass-dsl's own SM103 fix landed in **4.6.0** — but that same release
  also did an unrelated breaking API change (removed `cutlass.cute.core.ThrMma`)
  that vLLM's vendored flash-attention code was never updated for.
- vLLM's flash-attention import chain additionally pulls in a **third**
  package, `quack`, whose own cutlass-dsl compatibility is a third
  independently-moving variable. `quack` is what actually references
  `ThrMma` (`quack/layout_utils.py`), not vLLM's own code directly.

So: torch + vLLM + cutlass-dsl + quack all need to agree at once, and no
public combination does. This is normal for month-1-2 of a brand new
datacenter GPU, not a sign we're missing an obvious fix.

**Confirmed, don't re-relitigate:**
- The original venv (`a1x_rl/`: torch 2.8.0+cu128, vllm 0.11.0) cannot work
  on this hardware at all — CUDA 12.8 has zero kernels for `sm_103a`. This
  was the very first crash (Triton/ptxas failure at step 0).
- `vllm==0.20.0` is the **earliest** vLLM release whose published wheel is
  actually built against a CUDA-13-based torch (torch pin jumps
  2.9.1(cu12.8-default) → 2.10.0(cu12.8-default) → **2.11.0(cu13-default)**
  exactly at vllm 0.20.0). Versions 0.13.0–0.19.x DO have SM103 kernel
  *source* support (per vLLM's changelog), but their *published wheels*
  are still linked against CUDA 12 — installing them doesn't help; the
  compiled binary has no `sm_103a` cubins regardless of source support.
  **Do not downgrade vLLM** — we already verified this dead-ends.
- Building an older vLLM (0.14-0.19) from source against CUDA 13 was
  considered and rejected: unverified payoff (unknown which version each
  interface rename landed at), real risk of latent kernel bugs (SM103 was
  likely PTX-cross-compiled rather than hardware-validated by vLLM's CI
  at that vintage), and multi-hour compile cost. A full CUDA 13 `nvcc`
  toolchain does exist on this box (`/mnt/efs/users/rod/cuda-13.0`,
  `/opt/pytorch/cuda`) if this ever needs revisiting.
- An NGC container was considered: the freshest one available (rel-26-07)
  pins vLLM 0.24.0 — *newer* than what we're on, so switching would add
  more API churn, not less. Not investigated further.

## Environment

- Working venv: `recipes/alpamayo1_x_rl/a1x_rl_b300/` — torch `2.11.0+cu130`,
  vllm `0.20.0`, transformers `4.57.1` (unchanged from original pin).
- `recipes/alpamayo1_x_rl/env.sh` must be sourced before every launch. It
  now also sets `CUDA_HOME=/mnt/efs/users/rod/cuda-13.0` (see bug #5 below)
  and `WANDB_API_KEY`/`WANDB_ENTITY`.
- Launch command (from `recipes/alpamayo1_x_rl/`):
  `cosmos-rl --config recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test.toml --policy 1 --rollout 6 --log-dir /mnt/efs/users/rod/logs/ recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py`
- Logs land in `/mnt/efs/users/rod/logs/logs_<timestamp>/` — **do not trust
  the `logs_latest` symlink while a new run is starting**, it lags behind
  and points at the previous run's directory for a few seconds after
  relaunch. Match the directory by timestamp/mtime against the actual
  process start time instead.
- **Never `pip`/`uv pip install` into this venv while cosmos-rl processes
  are live.** The CUTE-DSL kernel paths do lazy/dynamic imports of
  `cutlass` submodules during actual kernel dispatch, not all upfront —
  live-patching the venv under a running process corrupts its import
  state (`cannot import name 'Int32' from 'cutlass'` was exactly this,
  not a real bug — always kill and relaunch after any pip change).

## Bugs found and fixed, in order hit

All fixes below are in the fork's own code (`recipes/alpamayo1_x_rl/models/reasoning_vla/`),
which owns a custom vLLM model plugin (`ReasoningVLAModelForVLLM` in
`vllm_wrapper.py`) wrapping `Qwen3VLForConditionalGeneration`. vLLM
explicitly documents its custom-model plugin interface as **not**
version-stable across releases — this whole bug class (#1-4) is the
ongoing cost of maintaining that plugin against vLLM 0.20.0, and will
recur on any future vLLM version bump, not just this one.

1. **`LLM(..., disable_mm_preprocessor_cache=...)` — kwarg removed.**
   vllm 0.20.0 replaced this bool with `mm_processor_cache_gb` (float GB
   size). Fixed in `rollout.py` (`mm_processor_cache_gb=0 if ... else 4`).

2. **`_reasoning_vla_vllm_hf_overrides(cfg)` crashed on `cfg.get_llm_config()`.**
   vllm 0.20.0 added a new dummy-probe call to `hf_overrides` with a bare
   `transformers.PretrainedConfig(architectures=[""], model_type="dummy_...")`
   before ever building the real config class, just to read `.model_type`.
   That bare object has no `get_llm_config`. Fixed in `rollout.py` with an
   `hasattr(cfg, "get_llm_config")` guard.

3. **`get_multimodal_embeddings`/`get_input_embeddings` never called.**
   vLLM 0.20's `SupportsMultiModal` protocol renamed these to
   `embed_multimodal`/`embed_input_ids`. The old names silently fell back
   to the Protocol's stub defaults (crash on every multimodal prompt, i.e.
   almost every prompt). Fixed in `vllm_wrapper.py`. The `embed_input_ids`
   fix initially missed a keyword-only `is_multimodal` parameter that
   vLLM's engine actually calls with (`TypeError: unexpected keyword
   argument 'is_multimodal'`) — fixed in a follow-up pass to match
   `Qwen3VLForConditionalGeneration.embed_input_ids`'s real signature
   exactly.

4. **Missing `SupportsMRoPE` interface.** `ReasoningVLAModelForVLLM` never
   declared this Protocol (`get_mrope_input_positions` + `supports_mrope`
   classvar), so `isinstance(model, SupportsMRoPE)` was `False` and
   `gpu_model_runner.py` asserted `"M-RoPE support is not implemented."`
   on every multimodal rollout, even though the wrapped Qwen3VL model
   genuinely supports it. Fixed in `vllm_wrapper.py` by adding
   `SupportsMRoPE` as a formal base class (matching vLLM's own pattern
   for `Qwen3VLForConditionalGeneration`) plus a delegating
   `get_mrope_input_positions`.

5. **`RuntimeError: DeepGEMM backend is not available or outdated.`**
   Not a cutlass-dsl issue — vLLM's vendored `vllm.third_party.deep_gemm`
   asserts `CUDA_HOME` is set during a kernel-warmup probe
   (`compile_or_warm_up_model` → `kernel_warmup` → `deep_gemm_warmup` →
   `_fp8_linear_may_use_deep_gemm` → `get_mk_alignment_for_contiguous_layout`)
   that runs unconditionally at engine init, **even for models that never
   use FP8** (this model's `quantization` is `"none"`). `CUDA_HOME` was
   simply never set in the launch environment. Fixed by adding
   `export CUDA_HOME=/mnt/efs/users/rod/cuda-13.0` to `env.sh`. (There is
   an identical CUDA 13.0 toolkit at `/opt/pytorch/cuda`, but that's on
   the instance's local/ephemeral NVMe disk; the EFS path is durable and
   matches prior TruckDrive-recipe convention.)

6. **`AssertionError: Only SM 10.x and 11.x are supported`** (vLLM's own
   vendored `vllm/vllm_flash_attn/cute/flash_fwd_sm100.py:142`). Root
   cause confirmed live in this venv: `nvidia-cutlass-dsl==4.5.3` was
   installed with its CUDA-12 native-lib variant
   (`nvidia-cutlass-dsl-libs-base`) even though torch is CUDA 13. This
   makes the `Arch.sm_110f` enum member silently alias `Arch.sm_101f`'s
   value, so the range check `Arch.sm_100 <= Arch.sm_103 <= Arch.sm_110f`
   wrongly evaluates `False` for this exact hardware. Fix: install the
   matching `nvidia-cutlass-dsl-libs-cu13` variant. **Must pin all three
   packages to the identical exact version** (`nvidia-cutlass-dsl`,
   `-libs-base`, `-libs-cu13`) — letting the resolver pick "latest" jumps
   to 4.7.0 and causes bug #7.

7. **`AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'`.**
   Direct consequence of fixing #6 by jumping to cutlass-dsl 4.7.0 (or any
   4.6.x): `ThrMma` was genuinely removed (not renamed — checked, not
   present anywhere in the package) from cutlass-dsl's API in the same
   4.6.0 release that fixed SM103. The crash is specifically in the
   ViT/multimodal-encoder attention path: `mm_encoder_attention.py` →
   `_forward_fa` → `vit_flash_attn_wrapper` → vLLM's vendored
   `vllm_flash_attn.cute` → imports the third-party `quack` package →
   `quack/layout_utils.py` references `ThrMma` at **import time**. Every
   published cutlass-dsl version was tested (4.5.3 through 4.7.0, plus
   all 4.6.x point releases) — none has both the SM103 fix and `ThrMma`
   intact; the two changes landed in the same version jump. **No pure
   version-pinning solution exists.**

   Fix: vLLM exposes `mm_encoder_attn_backend` (an `EngineArgs`/`LLM()`
   kwarg, accepts a string backend name) specifically to override the
   vision-encoder attention implementation. Set to `"FLASHINFER"` in
   `rollout.py`'s `LLM(...)` call to route around the broken
   `quack`-dependent flash-attn path entirely, while keeping cutlass-dsl
   at 4.7.0 (or any 4.6.x) for the SM103 fix everywhere else.

8. **`RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13`**
   (from `cudnn_frontend_shim.h`, used by FlashInfer's JIT-compiled
   kernels — surfaced immediately after fixing #7, once the vision
   encoder actually started using FlashInfer). Root cause:
   `nvidia-cutlass-dsl`'s own packaging makes `nvidia-cutlass-dsl-libs-cu12`
   an **unconditional** dependency (not gated behind the `cu13` extra) —
   installing the `cu13` extra adds cu13 *on top of*, not instead of, a
   full parallel cu12 native-lib stack (cublas, cudnn, cufft, curand,
   cusolver, cusparse, cusparselt, nvjitlink, nvtx, cuda-runtime, -nvrtc,
   -cupti, all `*-cu12`). `cudnn_frontend`'s native loader `dlopen()`s
   every candidate `libcudart.so.*` name and throws if more than one
   resolves. Fix: `uv pip uninstall` all the standalone `*-cu12` packages
   (confirmed via `importlib.metadata` that nothing else in the venv
   declares them as a dependency once cutlass-dsl's cu13 extra is
   installed).

   **While fixing this, also found and fixed unrelated filesystem
   corruption**: `nvidia/cusparselt/lib/libcusparseLt.so.0` was missing
   entirely, replaced by an orphaned NFS silly-rename artifact
   (`.nfs...`) — debris from the very first live-patch-while-running
   incident earlier in this migration (pip deleting a `.so` file that a
   live process still had open gets silently renamed to a hidden `.nfsXXX`
   placeholder on NFS instead of truly removed, and the real file never
   gets fully replaced). This broke `import torch` itself
   (`ImportError: libcusparseLt.so.0: cannot open shared object file`),
   unrelated to CUDA versioning. Fixed with
   `uv pip install --reinstall --no-deps nvidia-cusparselt-cu13==0.8.0`.
   There are more harmless orphaned `.nfs*` files sitting in
   `a1x_rl_b300/lib/python3.12/site-packages/nvidia/*/lib/` from the same
   incident (cublas, cudnn, cufft, curand, cusolver, cuda_runtime,
   cuda_nvrtc, cuda_cupti, nvjitlink, nvtx, cusparse) — inert disk-space
   cruft, not functional problems, safe to ignore or clean up.

   **Status as of this writing: fixes 1-8 all applied, venv verified to
   import torch/cutlass/flashinfer cleanly, relaunching to verify
   end-to-end.** If you're reading this because a run is still failing,
   check `ps aux` for whether the *current* run started before or after
   these fixes landed — if before, it's stale, kill it and relaunch.

## If you're a fresh instance picking this up

- Read this file in full before touching anything.
- Check `ps aux | grep -E "cosmos-rl|torchrun"` and the most recent
  `/mnt/efs/users/rod/logs/logs_<timestamp>/` directory (by mtime, not
  `logs_latest`) for current run status before assuming anything is
  broken or stuck.
- If you hit a *new* crash not listed above, the first move is always:
  find the exact file:line of the raise/assert, and check whether it's
  (a) in the fork's own `models/reasoning_vla/*.py` (our bug, fixable
  directly) or (b) inside `vllm`/`cutlass`/`quack`/`transformers`
  site-packages (likely a version-skew issue between those independently
  versioned packages — check exact installed versions and whether an
  override/config knob exists before assuming a code fix is needed).
