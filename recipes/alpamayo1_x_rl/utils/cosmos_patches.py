# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Runtime patches applied to Cosmos-RL from this fork.

Two unrelated groups, both monkeypatched rather than applied by editing the
venv, so a `uv sync` cannot silently revert them:

* :func:`apply_validation_accounting_patches` -- two upstream bugs that
  deadlock validation when `n_generation > 1`. Documented immediately below.
* :func:`apply_dynamic_sampling_patch` -- discards GRPO groups whose rewards
  have no spread, since upstream's equivalent is unreachable in this
  configuration. See that function's docstring.

---

Validation accounting: Cosmos-RL bugs that break validation when n_generation > 1.

Both bugs live in Cosmos-RL's validation accounting and both stem from the same
confusion: **prompts vs rollouts**. `val_datasize` counts prompts, but everything
compared against it counts rollouts, and there are `validation.n_generation`
rollouts per prompt. At `n_generation = 1` the two are equal and the bugs are
invisible, which is why upstream ships this way.

Observed 2026-08-06 with `[validation].n_generation = 6`: the run generated
rollouts forever and executed **zero** training steps.

---

**Bug 1 (fatal -- deadlock).** `cosmos_rl/dispatcher/status.py:633`:

    validation_finished = n_items_of_this_step == (
        self.data_fetcher.val_datasize or len(self.data_fetcher.val_dataloader)
    )

`n_items_of_this_step` sums *rollouts* (90 prompts x 6 = 540); `val_datasize` is
90 *prompts*. It is an exact `==`, so no value the counter passes through can
ever satisfy it. Consequences, in order: `clear_validation_status()` never runs,
so `activated_val_iter` stays live and validation prompts are re-dispatched in a
loop; the validation report is never emitted, so **no `val/*` metric ever
reaches W&B**; and `try_trigger_data_fetch_and_training()` -- the call that
starts training -- is never reached. Pending rollouts grow without a consumer.

**Bug 2 (cosmetic -- nonsense progress bar).** `cosmos_rl/dispatcher/status.py:637`:

    self.data_fetcher.activated_val_tqdm.update(n_items_of_this_step)

`n_items_of_this_step` is a *cumulative* total, but `tqdm.update(n)` *increments*
by `n`. Each report therefore adds the running total again and the bar grows
quadratically (sum of prefix sums). Verified: feeding cumulative 12,24,36,48
yields `tqdm.n = 120` where the true count is 48. This is what produced the
"validation: 804it" against a total of 90.

---

Fixes, both applied by monkeypatch rather than by editing the venv, so a
`uv sync` cannot silently revert them:

1. Scale `val_datasize` to rollouts (`len(val_set) * validation.n_generation`).
   This is the minimal change that makes Bug 1's `==` reachable, and it also
   puts the tqdm *total* into the same units as what is counted against it.
2. Wrap `activated_val_tqdm` so `update()` assigns absolutely instead of
   incrementing, which is what the call site actually means.

Remove this module if Cosmos-RL fixes the accounting upstream; the patches are
written to no-op safely rather than to fight a corrected implementation
(see `_ALREADY_APPLIED` and the `n_generation <= 1` guard).
"""

from __future__ import annotations

_ALREADY_APPLIED = False


class _AbsoluteUpdateTqdm:
    """Proxy making ``update(n)`` mean "position is now n" instead of "+= n".

    Everything other than ``update`` is forwarded to the real tqdm, so
    ``clear()`` / ``close()`` from ``clear_validation_status`` still work.
    """

    def __init__(self, inner):
        self._inner = inner

    def update(self, n=1):
        self._inner.n = n
        self._inner.refresh()

    def __getattr__(self, name):
        # Only reached for attributes not found normally, so `_inner` itself
        # resolves from __dict__ and this cannot recurse.
        return getattr(self._inner, name)


def apply_validation_accounting_patches() -> None:
    """Patch Cosmos-RL's prompt-vs-rollout validation accounting. Idempotent."""
    global _ALREADY_APPLIED
    if _ALREADY_APPLIED:
        return

    from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    # --- Bug 1: val_datasize is in prompts; it is compared against rollouts ---
    _orig_init = ControllerDataFetcher.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            n_gen = int(getattr(self.config.validation, "n_generation", 1) or 1)
        except AttributeError:
            return
        if not self.val_datasize or n_gen <= 1:
            return
        prompts = self.val_datasize
        self.val_datasize = prompts * n_gen
        logger.info(
            f"[alpamayo1_x_rl.patches] val_datasize {prompts} prompts -> "
            f"{self.val_datasize} rollouts (x{n_gen} generations). Without this, "
            "Cosmos-RL's `n_items == val_datasize` check can never fire and "
            "training never starts."
        )

    ControllerDataFetcher.__init__ = _patched_init

    # --- Bug 2: tqdm fed a cumulative value through an incrementing API ---
    _orig_activate = ControllerDataFetcher.validation_activate_dataloader

    def _patched_activate(self, validation_step: int):
        _orig_activate(self, validation_step)
        bar = self.activated_val_tqdm
        if bar is not None and not isinstance(bar, _AbsoluteUpdateTqdm):
            self.activated_val_tqdm = _AbsoluteUpdateTqdm(bar)

    ControllerDataFetcher.validation_activate_dataloader = _patched_activate

    _ALREADY_APPLIED = True
    logger.info("[alpamayo1_x_rl.patches] validation accounting patches applied")


# ---------------------------------------------------------------------------
# Dynamic sampling: drop groups whose rewards carry no learning signal
# ---------------------------------------------------------------------------

_DS_ALREADY_APPLIED = False

_DS_DEFAULTS: dict[str, float] = {
    # Groups whose reward std is below this are discarded. 0.0 disables.
    "min_group_reward_std": 0.01,
    # Never discard more than this fraction of the groups in one report, no
    # matter how many qualify -- see the stall discussion in the docstring.
    "max_drop_fraction": 0.5,
}


def _ds_cfg(config) -> dict[str, float]:
    """Read [custom.alpamayo.dynamic_sampling], falling back to defaults."""
    try:
        raw = getattr(config, "custom")["alpamayo"]["dynamic_sampling"]
    except (TypeError, KeyError, AttributeError):
        raw = {}
    out = {k: float(raw.get(k, v)) for k, v in _DS_DEFAULTS.items()}
    out["enable"] = bool(raw.get("enable", False))
    return out


def apply_dynamic_sampling_patch() -> None:
    """Discard zero-variance GRPO groups before they reach the training batch.

    **Why this is a fork patch and not a config flag.** Cosmos-RL ships DAPO
    dynamic sampling, but it cannot work here, for two independent reasons:

    1. `cosmos_rl/reward/local_calculator.py:165` and `:287` construct every
       `RLPayload` with a hardcoded `valid=True`. The consumer,
       `rollout_control.py:1459 dynamic_sampling()`, drops exactly those
       payloads whose `valid` is False. Nothing in the local reward path ever
       sets it False, so the filter is unreachable code.
    2. Even reached, it is gated behind `train_policy.variant == "dapo"` and
       keyed on `filter_reward_metric`, whose semantics are "rewards are all
       the same" -- exact equality. Our reward is continuous
       (`-traj_l2_weight * ade/ade_threshold + comfort_weight * comfort`, see
       rewards/aggregated_reward.py), so two completions are essentially never
       bit-identical and the equality test never fires. The degenerate case
       here is *near*-zero variance, not zero variance.

    So this patch filters on the group's reward **standard deviation** against
    a threshold instead.

    **Why dropping is worth doing.** GRPO's advantage is `r - mean(r)` over the
    group (`dispatcher/algo/grpo.py:38-46`; with `unbiased_advantage = true`
    there is no division by std). A group whose completions all score the same
    therefore contributes advantages of ~0 -- no gradient -- while still
    consuming a full share of the policy forward/backward, which is ~87% of
    step wall time. Worse, under `loss_type = "token-mean"`
    (`grpo_trainer.py:326-343`) the loss divides by the summed token count of
    the whole batch, so dead groups sit in the denominator and *shrink* the
    gradient contributed by the groups that did have signal. Dropping them
    both saves compute and stops them diluting the update.

    **Where it hooks in.** `LocalRewardCalculator.compute_rewards` is the last
    point where payloads are still grouped by prompt (one payload = one prompt
    = `n_generation` completions) and before anything is sent to the
    controller. Dropping here is safe with respect to batch assembly: the
    controller waits until `total_pending_rollouts() >= train_batch_per_replica
    * n_replicas` (`dispatcher/status.py:1202`) before triggering a step, so a
    short report just means the buffer fills from more prompts. The batch handed
    to the optimizer is still exactly `train_batch_per_replica`.

    **Two caveats, both deliberate.**

    * *Stall safety.* If nearly every group were degenerate, unconditional
      dropping could starve the buffer and hang the run. `max_drop_fraction`
      bounds this: within a single report we keep the highest-std groups up to
      that cap, so progress is always possible. It is a local, stateless rule.
    * *Step accounting drifts.* The controller's `remain_samples_num` is not
      decremented for what we drop, so steps-per-epoch is now an overestimate
      and the epoch boundary arrives sooner than the progress bar claims. This
      part really is cosmetic: `optm_decay_type` is unset, so the LR is constant
      and does not depend on `total_steps`.

    **!! KNOWN FATAL BUG -- this patch is DISABLED in every TOML as of
    2026-08-18. Do not re-enable without fixing this. !!**

    The paragraph above named the wrong counter and drew the wrong conclusion.
    `remain_samples_num` is cosmetic; **`samples_on_the_fly` is not**, and it
    leaks the same way:

        controller.py:430   dispatch  ->  += current_fetch_count * n_generation
        status.py:943       step done ->  -= train_batch_per_replica  (a CONSTANT)

    The decrement is a fixed constant rather than the number of rollouts
    actually consumed, so every group dropped here is `+n_generation` that is
    never subtracted, and the counter ratchets up monotonically instead of
    oscillating around zero.

    `controller.py:271` then throttles prompt dispatch once `samples_on_the_fly
    >= (allowed_outdated_steps + 1) * rollouts_per_global_batch`. On the motion
    run 20260817025604 that ceiling was 51 * 48 = 2448; the measured leak reached
    2360 by step 1003 and the run stopped dead there, after ~19h. The throttle is
    deliberately silent ("Log only when n is reduced but not when set to 0 since
    0 is logged too frequently"), so there is no warning, no error and no
    timeout: rollout workers spin in `request_new_prompts`, the policy spins in
    `broadcast_command` (its normal idle loop, so `py-spy` looks healthy), all
    GPUs stay allocated at 0% utilization, and no log says why.

    Note this cap is live under `[rollout].mode = "sync"` as well -- the
    throttle's gate is `config.mode == "colocated"`, the TOP-LEVEL mode field,
    not `[rollout].mode`. Raising `allowed_outdated_steps` only moves the stall
    later.

    **A fix cannot live entirely here.** `samples_on_the_fly` is controller
    state, and this patch runs in the rollout worker; the drop is invisible to
    the controller by construction. Any real fix has to make the drop count
    visible across that boundary -- e.g. reporting dropped rollouts to the
    controller so it can decrement, or moving the filter controller-side into
    the payload path where `RLPayload.valid` already exists (see reason 1 at the
    top of this docstring for why that flag is currently unreachable).

    Separately, before re-enabling: on that run it dropped 295 of 590 groups,
    exactly the `max_drop_fraction` cap, saturating on essentially every report.
    That discards half the rollout compute and systematically keeps the
    highest-variance groups every step -- a selection bias nothing in the loss
    accounts for. The feature assumes dead groups are occasional; at 50% that
    assumption is false, so `min_group_reward_std` is mistuned for this reward
    (or the reward genuinely lacks spread). Retune from a measured
    `dyn_sampling_group_std_mean` distribution collected with this OFF.

    Metrics, reported under `train/` in W&B (suffix conventions are
    `utils/util.py:1322-1344` -- `_count` sums, everything else means):

    * `dyn_sampling_groups_seen_count`    -- groups scored
    * `dyn_sampling_groups_dropped_count` -- groups discarded
    * `dyn_sampling_group_std_mean`       -- mean per-group reward std, over
      all groups *before* filtering. This is the number to watch: it says how
      much spread the reward function is actually producing, which is the
      thing that decides whether GRPO can learn at all here.

    Idempotent. Remove once upstream sets `valid` from a variance test.
    """
    global _DS_ALREADY_APPLIED
    if _DS_ALREADY_APPLIED:
        return

    import numpy as np

    from cosmos_rl.reward.local_calculator import (  # pyright: ignore[reportMissingImports]
        LocalRewardCalculator,
    )
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    _orig_compute_rewards = LocalRewardCalculator.compute_rewards

    def _patched_compute_rewards(self, payloads, is_validation, step):
        payload_list, is_val, out_step = _orig_compute_rewards(
            self, payloads, is_validation, step
        )
        # Validation must score every window it was given, or the val metrics
        # stop being comparable across steps.
        if is_val or not payload_list:
            return payload_list, is_val, out_step

        cfg = _ds_cfg(getattr(self, "config", None))
        # Announce the resolved settings once. `_ds_cfg` degrades to
        # enable=False when the TOML section cannot be read, and a silent
        # no-op is the worst outcome here: the run would look filtered and
        # not be. This makes the actual state greppable in rollout_*.log.
        if not getattr(self, "_alp_ds_logged", False):
            self._alp_ds_logged = True
            logger.info(
                f"[alpamayo1_x_rl.patches] dynamic sampling resolved config: {cfg} "
                "(enable=False here means [custom.alpamayo.dynamic_sampling] was "
                "missing or unreadable, NOT that filtering ran and found nothing)"
            )
        if not cfg["enable"] or cfg["min_group_reward_std"] <= 0.0:
            return payload_list, is_val, out_step

        stds = []
        for p in payload_list:
            rewards = p.rewards or []
            stds.append(float(np.std(rewards)) if len(rewards) > 1 else 0.0)

        threshold = cfg["min_group_reward_std"]
        # Rank by std so the cap keeps the most informative groups.
        order = sorted(range(len(payload_list)), key=lambda i: stds[i], reverse=True)
        max_drop = int(len(payload_list) * cfg["max_drop_fraction"])
        keep_idx, dropped_idx = [], []
        for i in order:
            if stds[i] < threshold and len(dropped_idx) < max_drop:
                dropped_idx.append(i)
            else:
                keep_idx.append(i)

        kept = [payload_list[i] for i in sorted(keep_idx)]
        if dropped_idx:
            capped = sum(1 for i in keep_idx if stds[i] < threshold)
            logger.info(
                f"[alpamayo1_x_rl.patches] dynamic sampling @ step {out_step}: "
                f"dropped {len(dropped_idx)}/{len(payload_list)} groups with "
                f"reward std < {threshold:g} "
                f"(group std mean {np.mean(stds):.4f}, max {np.max(stds):.4f})"
                + (
                    f"; kept {capped} degenerate group(s) to respect "
                    f"max_drop_fraction={cfg['max_drop_fraction']:g}"
                    if capped
                    else ""
                )
            )

        # The two suffixes aggregate differently (utils/util.py:1330-1344) and
        # each needs the opposite placement:
        #   `_count` -> np.sum over every rollout in the step. Put it on ONE
        #       rollout, or it gets multiplied by the number of rollouts.
        #   `_mean`  -> np.mean over every rollout in the step, with
        #       `data.get(k, 0)` for rollouts that lack the key. Absent is
        #       therefore not "skipped" but "zero", so it must go on EVERY
        #       rollout or the mean is scaled down by the coverage fraction.
        std_mean = float(np.mean(stds))
        for p in kept:
            for m in p.report_metrics or []:
                m["dyn_sampling_group_std_mean"] = std_mean
        if kept and kept[0].report_metrics:
            kept[0].report_metrics[0].update(
                {
                    "dyn_sampling_groups_seen_count": len(payload_list),
                    "dyn_sampling_groups_dropped_count": len(dropped_idx),
                }
            )
        return kept, is_val, out_step

    LocalRewardCalculator.compute_rewards = _patched_compute_rewards

    _DS_ALREADY_APPLIED = True
    logger.info("[alpamayo1_x_rl.patches] dynamic sampling patch applied")


# ---------------------------------------------------------------------------
# Log train/epoch alongside the step-indexed metrics
# ---------------------------------------------------------------------------

_EPOCH_ALREADY_APPLIED = False


def apply_epoch_logging_patch() -> None:
    """Add ``train/epoch`` to every W&B report. Idempotent.

    Cosmos-RL reports ``train_step`` and ``train/total_steps`` but never an
    epoch, while the schedule (``[train].epoch``) and every decision we make
    about it are expressed in epochs. Reading "peaked at step 600" requires
    dividing by a steps-per-epoch figure that appears nowhere in the logs.

    Derivation, and its one assumption: at startup Cosmos-RL sets
    ``total_steps = epoch * steps_per_epoch`` (dispatcher/status.py:243-272
    ``recompute_total_steps``), so ``steps_per_epoch = total_steps / epoch``.
    That is captured from the FIRST report and then held fixed, because
    ``total_steps`` is recomputed from remaining samples as the run proceeds
    and drifts downward -- notably because our own dynamic sampling discards
    groups without decrementing the controller's sample accounting. Recomputing
    the divisor every step would make the epoch axis shrink as the run went on.

    Hooked at ``log_wandb`` rather than at the report-building code because
    every metric path (train at status.py:1098, validation at :845, :688) funnels
    through it, so train and val curves both get the field with one patch.
    Note ``status.py`` does ``from ... import log_wandb`` at module level, so
    the name must be rebound on the *status* module -- patching
    ``utils.report.wandb_logger`` would not affect the already-bound reference.

    Falls back to leaving the data untouched if anything is missing, since a
    metrics annotation must never be able to take down a training run.
    """
    global _EPOCH_ALREADY_APPLIED
    if _EPOCH_ALREADY_APPLIED:
        return

    import cosmos_rl.dispatcher.status as _status  # pyright: ignore[reportMissingImports]
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    _orig_log_wandb = _status.log_wandb
    state: dict[str, float] = {}

    def _steps_per_epoch(data: dict) -> float | None:
        if "steps_per_epoch" in state:
            return state["steps_per_epoch"]
        total_steps = data.get("train/total_steps")
        if not total_steps:
            return None
        epochs = _configured_epochs()
        if not epochs:
            return None
        spe = float(total_steps) / float(epochs)
        if spe <= 0:
            return None
        state["steps_per_epoch"] = spe
        logger.info(
            f"[alpamayo1_x_rl.patches] epoch logging: {spe:.1f} steps/epoch "
            f"(total_steps {total_steps} / epoch {epochs}); train/epoch now reported"
        )
        return spe

    def _patched_log_wandb(data: dict, step: int):
        try:
            spe = _steps_per_epoch(data)
            if spe:
                data = {**data, "train/epoch": step / spe}
        except Exception as e:  # pragma: no cover - never break logging
            logger.warning(f"[alpamayo1_x_rl.patches] epoch logging skipped: {e}")
        return _orig_log_wandb(data=data, step=step)

    _status.log_wandb = _patched_log_wandb

    _EPOCH_ALREADY_APPLIED = True
    logger.info("[alpamayo1_x_rl.patches] epoch logging patch applied")


# ---------------------------------------------------------------------------
# Reward-time step capture: give traj_viz the real weight step, not an estimate
# ---------------------------------------------------------------------------

_STEP_CAPTURE_ALREADY_APPLIED = False


def apply_reward_step_capture_patch() -> None:
    """Capture the true ``(step, is_validation)`` Cosmos-RL already has at
    reward time, for :mod:`alpamayo1_x_rl.rewards.traj_viz` to read instead of
    approximating its own. Idempotent.

    **The problem.** The reward function (``compute_reward``/``compute_group_reward``
    in ``rewards/aggregated_reward_with_reasoning.py``) runs in the ROLLOUT
    replica and is handed only ``to_be_evaluated``, ``reference``, ``prompt``,
    ``data_packer``, ``config``, ``tokenizer`` --
    ``cosmos_rl/dispatcher/algo/reward.py``'s ``compute_per_reward_func`` never
    forwards a step or a validation flag. Lacking that, ``traj_viz.py`` used to
    derive its own x-axis from a reward-call counter (``_CALL_COUNT //
    train_batch_per_replica``). That estimate has no way to know a given call
    came from validation rather than training, so a validation round inflates
    it exactly as if it were extra training steps. Confirmed on run
    20260817025710: `viz/step` jumped 0 -> 214 across one validation round
    (528 rollouts, matching `val/rollout_count`), while the run's real
    `val/step` only moved 0 -> 150 in that span.

    **The fix.** ``LocalRewardCalculator.compute_rewards(self, payloads,
    is_validation, step)`` is the one place in the rollout process that already
    has both pieces of ground truth -- its own docstring calls ``step`` "the
    weight step where the payloads are generated", i.e. the same quantity
    Cosmos-RL reports as ``train_step``/``val/step``. This patch stashes both,
    per-thread, right before that call dispatches to the reward function
    (``RolloutGroup.compute_rollouts`` inside it, synchronously).

    **Why thread-local, not a plain module global.** `RewardDispatcher` runs
    reward calculation on a `ThreadPoolExecutor` when `[train].non_text = true`
    (this recipe's setting) with `num_workers` possibly > 1, and all worker
    threads share the one `LocalRewardCalculator` singleton this patches. A
    bare global would let a validation batch on one thread clobber a training
    batch's flag on another mid-call; `threading.local()` keeps each in-flight
    `compute_rewards` call's state isolated to the thread that owns it, which
    is also the thread that goes on to call the reward function synchronously.
    """
    global _STEP_CAPTURE_ALREADY_APPLIED
    if _STEP_CAPTURE_ALREADY_APPLIED:
        return

    from cosmos_rl.reward.local_calculator import (  # pyright: ignore[reportMissingImports]
        LocalRewardCalculator,
    )
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from alpamayo1_x_rl.rewards import traj_viz

    _orig_compute_rewards = LocalRewardCalculator.compute_rewards

    def _patched_compute_rewards(self, payloads, is_validation, step):
        traj_viz.set_current_weight_step(step, is_validation)
        return _orig_compute_rewards(self, payloads, is_validation, step)

    LocalRewardCalculator.compute_rewards = _patched_compute_rewards

    _STEP_CAPTURE_ALREADY_APPLIED = True
    logger.info("[alpamayo1_x_rl.patches] reward step-capture patch applied")


def _configured_epochs() -> int | None:
    """Read ``[train].epoch`` from the TOML named by ``--config`` in argv.

    Mirrors ``launcher._read_ckpt_path_from_toml``; the controller is launched
    with a rewritten temp TOML, but ``[train].epoch`` is carried through
    unchanged.
    """
    import sys
    import tomllib

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            try:
                with open(sys.argv[i + 1], "rb") as f:
                    return int(tomllib.load(f).get("train", {}).get("epoch") or 0) or None
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Dataset provenance: put the resolved dataset selection into the W&B config
# ---------------------------------------------------------------------------

_DSCFG_ALREADY_APPLIED = False


def apply_dataset_config_logging_patch() -> None:
    """Record which dataset a run used in its W&B config. Idempotent.

    **The gap.** Cosmos-RL snapshots the run's config with
    ``config=config.model_dump()`` (``utils/report/wandb_logger.py:88``), which
    covers the TOML -- including ``[custom.alpamayo]``, since ``custom`` is part
    of the model. But this fork does not put the dataset in the TOML: the
    ``[train.train_policy.dataset]`` fields are left blank and the real
    selection is injected as Hydra overrides resolved from env vars at entry
    point import (see ``dataset_spec.py``). Hydra configures a separate Alpamayo
    object that Cosmos-RL never sees, and env vars are not in the dump either.

    Net effect: the single most important fact about a run -- what it trained on
    -- was absent from its permanent record. ``launcher._write_dataset_manifest``
    writes ``config_dataset.txt`` into ``output_dir``, but that lives on
    ``/opt/dlami/nvme`` (ephemeral instance storage), so it dies with the box
    while the W&B run outlives it.

    **The fix.** Merge ``dataset_spec.LAST_RESOLVED.summary`` into
    ``config.custom["alpamayo"]["dataset"]`` before ``init_wandb`` dumps it, so
    the selection rides along in the config W&B already captures. No new logging
    path, and it shows up in the config panel and in run comparisons/filters.

    **Why patch the controller's name, not the logger module.**
    ``dispatcher/controller.py:33`` does ``from ...wandb_logger import
    init_wandb`` at module level, binding the function object into the
    controller's namespace at import. Rebinding ``wandb_logger.init_wandb``
    would leave that reference pointing at the original -- the same trap noted
    for ``log_wandb`` in the epoch patch. Both names are rebound here so the
    patch holds whichever way the call is reached.

    Best-effort by construction: any failure leaves the original behaviour
    intact, because a provenance annotation must never be able to fail a run.
    """
    global _DSCFG_ALREADY_APPLIED
    if _DSCFG_ALREADY_APPLIED:
        return

    import cosmos_rl.dispatcher.controller as _controller  # pyright: ignore[reportMissingImports]
    import cosmos_rl.utils.report.wandb_logger as _wandb_logger  # pyright: ignore[reportMissingImports]
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    _orig_init_wandb = _wandb_logger.init_wandb

    def _patched_init_wandb(config):
        try:
            import alpamayo1_x_rl.dataset_spec as dataset_spec

            resolved = dataset_spec.LAST_RESOLVED
            if resolved is not None:
                custom = getattr(config, "custom", None)
                if isinstance(custom, dict):
                    custom.setdefault("alpamayo", {})["dataset"] = resolved.summary
                    logger.info(
                        "[alpamayo1_x_rl.patches] dataset provenance -> W&B config: "
                        f"{resolved.summary}"
                    )
                else:
                    logger.warning(
                        "[alpamayo1_x_rl.patches] config.custom is not a dict "
                        f"({type(custom).__name__}); dataset provenance NOT recorded "
                        "in W&B. The run is unaffected."
                    )
        except Exception as e:  # pragma: no cover - never block a run
            logger.warning(
                f"[alpamayo1_x_rl.patches] dataset provenance skipped: {e}"
            )
        return _orig_init_wandb(config)

    _wandb_logger.init_wandb = _patched_init_wandb
    if getattr(_controller, "init_wandb", None) is not None:
        _controller.init_wandb = _patched_init_wandb

    _DSCFG_ALREADY_APPLIED = True
    logger.info("[alpamayo1_x_rl.patches] dataset config logging patch applied")
