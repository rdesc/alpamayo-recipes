# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

"""Phase 2, Experiment A: paired sampling displacement of the deployed flow-matching sampler.

Phase 1 asked whether the reasoning changes the DENSITY at the ground-truth trajectory, and
answered no (see ``FM_HEAD_STATUS.md``). That is one number at one point of a 128-dimensional
space. The deployed system does not evaluate a density -- it SAMPLES. Two conditionings can put
the same density on ``a*`` and still push the sampler somewhere else.

So here we run the sampler itself, twice, from the SAME initial noise:

    x_0 ~ N(0, I);  x_{k+1} = x_k + dt_k * v_theta(x_k, t_k, KV(vision, coc))

(``alpamayo_r1/diffusion/flow_matching.py::_euler``: ``t = linspace(0, 1, 11)``, so K=10 and
dt=0.1. The ``linspace`` is reproduced here rather than hard-coding 0.1, so the step sizes are
bit-identical to deployment.)

Common random numbers across arms is the whole design: with x_0 shared, every metre of
displacement between two rollouts is attributable to the conditioning, not to the draw.

    displacement(arm) = ADE( traj(rollout(x_0, gold)), traj(rollout(x_0, arm)) )   [metres]

ARMS
    null         gold prose scored twice, re-running the VLM prefill from scratch. MUST be
                 exactly 0 -- it is the determinism check, not a noise floor.
    donor        the headline: another clip's chain-of-causation, same vision.
    cluster      a donor from the SAME ``event_cluster``: topic held, scene varied.
    shuffled     this event's OWN words permuted. Same bag of words, destroyed syntax.
    blankvision  the yardstick: cameras zeroed, gold prose kept. `wrongtraj` is meaningless in
                 this experiment -- it varies the TARGET, and a sampler has no target.

``donor``/``cluster``/``shuffled`` were statistically indistinguishable on the likelihood in
Phase 1 -- that indistinguishability IS the occupancy result. Carrying all three here is what
makes it possible to say whether they stay indistinguishable once the sampler runs.

Two further yardsticks come free from the gold rollouts themselves and are recorded per event,
because "displacement in metres" means nothing without them:

    spread_gold  mean pairwise ADE among the S gold rollouts. How far the sampler moves the car
                 when NOTHING changes but the dice. If swapping the prose moves it less than
                 re-rolling the dice does, the reasoning is behaviourally inert at deployment.
    ade_gold_gt  each gold rollout against the true future -- the model's own accuracy scale.

Cost: 10 velocity evaluations per arm (batched over draws), no Jacobians, no inversion, no
Newton. Seconds per event against ~24 min for the exact density of Phase 1.

Usage (from recipes/alpamayo1_x_rl, with that recipe's venv -- see the root CLAUDE.md):

    ./a1x_rl_b300/bin/python .../phase2_sample_displacement.py \
        --shard-idx 0 --num-shards 6 --n-draws 16 \
        --out /mnt/efs/users/rod/results/llr_fm_sampdisp/samp.shard0.parquet
"""

import argparse
import contextlib
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path[:0] = ["../../../src", "../.."]

import pandas as pd
import physical_ai_av
from alpamayo_r1.helper import to_device
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo.processor.qwen_processor import (
    build_processor,
    collate_fn_from_model_config,
    get_preprocess_data_fn_from_model_config,
)
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shard-idx", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--split", default="both",
                help="Phase 1's headline ladder ran on `both` (n=2,071); the exact discrete-map\n"
                     "arm ran on `val` (n=208). `both` here so per-event displacement can be\n"
                     "joined against the Phase-1 per-event LLR.")
ap.add_argument("--steps", type=int, default=10,
                help="MUST match num_inference_steps (10) to be the deployed map")
ap.add_argument("--n-draws", type=int, default=16,
                help="Initial-noise draws per event, shared across arms (common random numbers).")
ap.add_argument("--arms", default="null,donor,cluster,shuffled,blankvision")
ap.add_argument("--donor-seed", type=int, default=0)
ap.add_argument("--noise-seed", type=int, default=0,
                help="Seeds the per-event x_0 draw. Unlike the Phase-1 surrogate -- whose eps "
                     "grid was drawn ONCE outside the event loop, so its Monte-Carlo error did "
                     "not shrink as 1/sqrt(n) -- the draw here is derived per event from "
                     "(noise_seed, clip_id, event_idx), so grid error is independent across "
                     "events and averages out.")
ap.add_argument("--fp32", action="store_true",
                help="Precision control: VLM fp32 on cuda:0, expert fp32 on cuda:1. Two GPUs per "
                     "worker. The default (bf16 everywhere, one GPU) is what the car actually "
                     "runs, so it is the faithful measurement; this arm bounds the numerical "
                     "contribution on a paired subset. Full fp32 on ONE GPU OOMs at 79 GB.")
ap.add_argument("--save-traj", action="store_true", default=True,
                help="Store the rolled-out (64, 2) xy trajectories per draw. ~40 kB/event, and "
                     "it is what Experiment B's direction projection will need.")
ap.add_argument("--out", required=True)
args = ap.parse_args()

# ---------------- events ----------------
df = pd.read_parquet(PARQUET)
if args.split != "both":
    df = df[df["split"] == args.split].copy()
df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
df = df.dropna(subset=["events"])
events = []
for cid, row in df.iterrows():
    for ei, ev in enumerate(row["events"]):
        events.append(dict(clip_id=cid, event_idx=ei,
                           t0_us=max(int(ev["event_start_timestamp"]), MIN_T0_US),
                           gold_coc=ev["coc"], event_cluster=row["event_cluster"],
                           split=row["split"]))
donor_pool = [(e["clip_id"], e["gold_coc"], e["event_cluster"]) for e in events]
sel = events[args.shard_idx::args.num_shards]
if args.limit:
    sel = sel[:args.limit]
arms = [a for a in args.arms.split(",") if a]
for a in arms:
    if a not in ("null", "donor", "cluster", "shuffled", "blankvision"):
        raise SystemExit(f"unknown arm {a!r} (wrongtraj is meaningless here -- see the docstring)")
print(f"[sampdisp] {len(sel)} events, arms={arms}, steps={args.steps}, draws={args.n_draws}, "
      f"dtype={'fp32' if args.fp32 else 'bf16'}", flush=True)

# ---------------- model ----------------
# DEFAULT IS bf16 EVERYWHERE, and that is deliberate -- the opposite call from the Phase-1 exact
# density. There the object of study was "the density of the 10-step map with the velocity field
# evaluated exactly", so the expert ran fp32 (bf16 rounding is not differentiable and the
# log-determinant accumulates). Here the object of study is what the car DOES, and the car runs
# bf16. Rounding is deterministic given identical inputs, so the `null` arm still has to come out
# at exactly 0; --fp32 exists to check on a paired subset that the donor displacement is not an
# artefact of bf16 chaos amplifying a numerically tiny KV difference.
#
# `torch_dtype=` is silently ignored by this constructor (config.model_dtype=bfloat16 with
# keep_same_dtype=True casts the expert regardless while ~half the VLM stays fp32, which raises
# "mat1 and mat2 must have the same dtype"). Cast explicitly AFTER construction.
DT = torch.float32 if args.fp32 else torch.bfloat16
VDEV, EDEV = ("cuda:0", "cuda:1") if args.fp32 else ("cuda:0", "cuda:0")
model = TrainableAlpamayoR1.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
model.vlm.to(device=VDEV, dtype=DT)
for mod in (model.expert, model.action_in_proj, model.action_out_proj):
    mod.to(device=EDEV, dtype=DT)
model.action_space.to(device=EDEV)
_edt = {str(q.dtype) for q in model.expert.parameters()}
_vdt = {str(q.dtype) for q in model.vlm.parameters()}
assert _edt == {str(DT)}, f"mixed expert dtypes: {_edt}"
assert _vdt == {str(DT)}, f"mixed vlm dtypes: {_vdt}"
print(f"[sampdisp] vlm={DT} on {VDEV} ({torch.cuda.memory_allocated(VDEV) / 2**30:.1f} GiB)  "
      f"expert={DT} on {EDEV} ({torch.cuda.memory_allocated(EDEV) / 2**30:.1f} GiB)", flush=True)
adims = tuple(model.action_space.get_action_space_dims())
fstart = model.config.traj_token_ids["future_start"]
pre = get_preprocess_data_fn_from_model_config(
    components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
    components_prompt=["cot", "traj_future"], label_components=["cot"],
    generation_mode=False, include_camera_ids=True, include_frame_nums=True,
    model_config=model.config, chat_template_version="r1")
build_processor(vlm_name_or_path=model.config.vlm_name_or_path,
                traj_vocab_size=model.config.traj_vocab_size,
                min_pixels=model.config.min_pixels, max_pixels=model.config.max_pixels,
                include_camera_ids=True, include_frame_nums=True, chat_template_version="r1")
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
print(f"[sampdisp] adims={adims}", flush=True)


def kv_for(data, coc_text, blank_vision=False):
    """VLM prefill for one conditioning, cropped at <traj_future_start>. Same path as the
    Phase-1 scorers, so the two experiments condition on byte-identical caches."""
    s = {"image_frames": (torch.zeros_like(data["image_frames"]) if blank_vision
                          else data["image_frames"]),
         "camera_indices": data["camera_indices"],
         "relative_timestamps": data["relative_timestamps"],
         "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
         "ego_history_rot": data["ego_history_rot"].squeeze(0),
         "cot": coc_text}
    s["tokenized_data"] = pre(s)
    batch = collate_fn_from_model_config([dict(s)], model_config=model.config,
                                         include_camera_ids=True, include_frame_nums=True,
                                         chat_template_version="r1")
    td = to_device({"td": dict(batch["tokenized_data"])}, VDEV)["td"]
    ids = td.pop("input_ids")
    ids = model.fuse_traj_tokens(ids, {k: data[k].to(VDEV) for k in
                                       ("ego_history_xyz", "ego_history_rot",
                                        "ego_future_xyz", "ego_future_rot")})
    with torch.no_grad():
        out = model.vlm.model(input_ids=ids, use_cache=True, **td)
    fs = (ids == fstart).nonzero(as_tuple=False)
    assert fs.shape[0] == 1, f"expected one <traj_future_start>, got {fs.shape[0]}"
    cache = out.past_key_values
    plen = int(fs[0, 1].item()) + 1
    cache.crop(plen)
    delta_out = int(out.rope_deltas.flatten()[0].item()) + int(cache.get_seq_length())
    snap = [(l.keys.to(device=EDEV, dtype=DT), l.values.to(device=EDEV, dtype=DT))
            for l in cache.layers]
    del cache, out
    return snap, delta_out, plen


def make_vfield(snap, delta):
    """The deployed step_fn (alpamayo_r1/models/alpamayo_r1.py::step_fn), minus the padding
    attention mask -- there is one sequence per prefill here, so nothing is padded."""
    def vfield(x, t):
        B = x.shape[0]
        cache = DynamicCache(ddp_cache_data=[
            (k.expand(B, *k.shape[1:]), v.expand(B, *v.shape[1:])) for k, v in snap])
        tt = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(1).expand(B)
        tt = tt.reshape(B, *([1] * (x.dim() - 1)))
        emb = model.action_in_proj(x, tt)
        pos = (torch.arange(emb.shape[1], device=emb.device)
               .view(1, 1, -1).expand(3, B, -1).clone() + delta)
        fwd = {"is_causal": False} if model.config.expert_non_causal_attention else {}
        eo = model.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache,
                          attention_mask=None, use_cache=False, **fwd)
        return model.action_out_proj(eo.last_hidden_state[:, -emb.shape[1]:]).view(B, *adims)
    return vfield


def euler_rollout(x0, vfield, K):
    """The deployed sampler, verbatim: ``torch.linspace(0, 1, K+1)`` and explicit Euler."""
    ts = torch.linspace(0.0, 1.0, K + 1, device=x0.device, dtype=x0.dtype)
    x = x0
    with torch.no_grad():
        for i in range(K):
            x = x + (ts[i + 1] - ts[i]) * vfield(x, ts[i])
    return x


def action_to_xy(action, data):
    """(S, 64, 2) action -> (S, 64, 2) future xy in METRES, ego frame."""
    S = action.shape[0]
    hx = data["ego_history_xyz"].to(device=EDEV, dtype=torch.float32)
    hr = data["ego_history_rot"].to(device=EDEV, dtype=torch.float32)
    hx = hx.reshape(1, *hx.shape[-2:]).expand(S, -1, -1)
    hr = hr.reshape(1, *hr.shape[-3:]).expand(S, -1, -1, -1)
    with torch.no_grad():
        xyz, _ = model.action_space.action_to_traj(action.float(), hx, hr)
    return xyz[..., :2].double().cpu().numpy()


def gold_action(data):
    return model.action_space.traj_to_action(
        traj_history_xyz=data["ego_history_xyz"].to(EDEV),
        traj_history_rot=data["ego_history_rot"].to(EDEV),
        traj_future_xyz=data["ego_future_xyz"].to(EDEV),
        traj_future_rot=data["ego_future_rot"].to(EDEV),
    ).reshape(1, *adims).to(device=EDEV, dtype=DT)


def _seed(tag, cid, ei):
    """Deterministic per (tag, event). hashlib, not hash(): CPython salts str hashing per
    process, so hash() would give different draws in every shard and re-run."""
    key = f"{args.donor_seed}|{tag}|{cid}|{int(ei)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _rng(tag, cid, ei):
    return np.random.default_rng(_seed(tag, cid, ei))


def draw_x0(cid, ei):
    """Per-event initial noise, identical across arms. Drawn on CPU with an explicit generator so
    it does not depend on the GPU RNG stream (which the VLM prefill would otherwise advance)."""
    key = f"{args.noise_seed}|x0|{cid}|{int(ei)}".encode()
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    z = torch.randn(args.n_draws, *adims, generator=g, dtype=torch.float32)
    return z.to(device=EDEV, dtype=DT)


def shuffle_words(gold, cid, ei):
    words = gold.split()
    if len(words) < 3:
        return None
    r = _rng("shuffled", cid, ei)
    w = list(words)
    for _ in range(10):  # a permutation equal to the original is not a control
        r.shuffle(w)
        if w != words:
            break
    return " ".join(w)


def _ade(a, b):
    """Mean over waypoints of the per-waypoint euclidean distance. (S,) for (S,64,2) inputs."""
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


rows, t0run = [], time.time()
for i, ev in enumerate(sel):
    try:
        data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
        cid, ei = ev["clip_id"], ev["event_idx"]
        x0 = draw_x0(cid, ei)
        gt_xy = np.asarray(data["ego_future_xyz"].detach().cpu()).reshape(-1, 3)[:, :2]

        conds = [("gold", ev["gold_coc"], dict())]
        if "null" in arms:
            conds.append(("null", ev["gold_coc"], dict()))
        if "donor" in arms:
            r = _rng("random", cid, ei)
            cands = [j for j, (c, _, _) in enumerate(donor_pool) if c != cid]
            conds.append(("donor", donor_pool[cands[int(r.integers(len(cands)))]][1], dict()))
        if "cluster" in arms:
            r = _rng("cluster", cid, ei)
            cands = [j for j, (c, _, cl) in enumerate(donor_pool)
                     if c != cid and cl == ev["event_cluster"]]
            if cands:
                conds.append(("cluster",
                              donor_pool[cands[int(r.integers(len(cands)))]][1], dict()))
        if "shuffled" in arms:
            sh = shuffle_words(ev["gold_coc"], cid, ei)
            if sh is not None:
                conds.append(("shuffled", sh, dict()))
        if "blankvision" in arms:
            conds.append(("blankvision", ev["gold_coc"], dict(blank=True)))

        ctx = (contextlib.nullcontext() if args.fp32
               else torch.autocast("cuda", dtype=torch.bfloat16))
        xy, plens = {}, {}
        for name, txt, opt in conds:
            snap, delta, plen = kv_for(data, txt, blank_vision=opt.get("blank", False))
            vf = make_vfield(snap, delta)
            t0 = time.time()
            with ctx, sdpa_kernel([SDPBackend.MATH]):
                act = euler_rollout(x0, vf, args.steps)
            xy[name] = action_to_xy(act, data)
            plens[name] = (plen, time.time() - t0)
            del snap, vf, act
            torch.cuda.empty_cache()

        g = xy["gold"]
        # sampler's own diversity: mean pairwise ADE among the gold rollouts, the honest
        # denominator for "did the prose move the car"
        pw = [float(_ade(g[a:a + 1], g[b:b + 1])[0])
              for a in range(len(g)) for b in range(a + 1, len(g))]
        spread = float(np.mean(pw)) if pw else float("nan")
        ade_gt = _ade(g, gt_xy[None, :, :])

        for name, _, _ in conds:
            d = _ade(xy[name], g)
            # displacement says the two rollouts DIFFER; accuracy says whether the difference is
            # a degradation. They are separate questions and only the second speaks to whether a
            # regulariser would have anything to reinforce, so both are recorded per arm.
            acc = _ade(xy[name], gt_xy[None, :, :])
            rec = dict(clip_id=cid, event_idx=ei, event_cluster=ev["event_cluster"],
                       split=ev["split"], cond=name,
                       ade=d.astype(np.float64),
                       fde=np.linalg.norm(xy[name][:, -1] - g[:, -1], axis=-1).astype(np.float64),
                       ade_mean=float(d.mean()), n_draws=int(args.n_draws),
                       ade_gt=acc.astype(np.float64), ade_gt_mean=float(acc.mean()),
                       ade_gt_min=float(acc.min()),
                       prefix_len=plens[name][0], secs=plens[name][1],
                       spread_gold=spread, ade_gold_gt_mean=float(ade_gt.mean()),
                       ade_gold_gt_min=float(ade_gt.min()))
            if args.save_traj:
                rec["traj_xy"] = xy[name].astype(np.float32).reshape(-1)
                rec["gt_xy"] = gt_xy.astype(np.float32).reshape(-1)
            rows.append(rec)

        msg = " ".join(f"{n}={float(_ade(xy[n], g).mean()):.3f}m"
                       f"/{float(_ade(xy[n], gt_xy[None, :, :]).mean()):.2f}gt"
                       for n, _, _ in conds if n != "gold")
        print(f"[sampdisp] {i + 1}/{len(sel)} {str(cid)[:8]} spread={spread:.3f}m "
              f"gt={ade_gt.mean():.3f}m | {msg} | {time.time() - t0run:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001 -- one bad clip must not sink a 6-hour shard
        print(f"[sampdisp] SKIP {ev['clip_id']} idx={ev['event_idx']}: "
              f"{type(e).__name__}: {e}", flush=True)
        continue

    if (i + 1) % 20 == 0:
        pd.DataFrame(rows).to_parquet(args.out)

pd.DataFrame(rows).to_parquet(args.out)
print(f"[sampdisp] wrote {args.out} rows={len(rows)}", flush=True)
