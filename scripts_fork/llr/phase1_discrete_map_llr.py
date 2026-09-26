# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Exact likelihood of the DEPLOYED 10-step Euler sampler, on real events.

Validation arm for the reweighted-ELBO surrogate. The surrogate differences two variational
BOUNDS and the slack need not cancel; this differences two EXACT log-densities of the map the
car actually runs, so it has no slack at all. See fm_llr_method.md section 9.

Arms mirror the surrogate's ladder so the ratios are directly comparable:
    gold / donor (random CoC) / wrongtraj (different target) / blankvision (zeroed frames)
"""
import os, sys, json, time, argparse, math, hashlib, contextlib
import numpy as np, torch
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_map_core import logp_discrete_map
sys.path[:0] = ["../../../src", "../.."]

import pandas as pd
import physical_ai_av
from alpamayo_r1.helper import to_device
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo.processor.qwen_processor import (
    build_processor, collate_fn_from_model_config, get_preprocess_data_fn_from_model_config)
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shard-idx", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--split", default="val")
ap.add_argument("--steps", type=int, default=10, help="MUST match num_inference_steps to be the deployed map")
ap.add_argument("--tchunk", type=int, default=16)
ap.add_argument("--arms", default="donor,wrongtraj,blankvision")
ap.add_argument("--donor-seed", type=int, default=0)
ap.add_argument("--donor-traj-min-ade", type=float, default=5.0,
                help="Minimum ADE (m, 6.4 s) between this event's GT future and the wrongtraj "
                     "donor. Matches the token head's gate (phase0_llr_sanity_controls.py) so "
                     "the separation -- the denominator both heads normalise by -- is the SAME "
                     "measurement on both. Ungated, ~92%% of this split is 'Continue straight' "
                     "and the 'wrong' target is often a near-duplicate. 0 disables.")
ap.add_argument("--donor-traj-max-tries", type=int, default=32)
ap.add_argument("--bf16", action="store_true", help="debug only; the whole point of this arm is fp32")
ap.add_argument("--device-split", action="store_true",
                help="VLM on cuda:0, expert on cuda:1, BOTH fp32. Needs 2 GPUs per worker but "
                     "removes the bf16 KV cache, the last precision caveat on the donor contrast. "
                     "(NB not --split, which is the DATASET split.)")
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
                           gold_coc=ev["coc"], event_cluster=row["event_cluster"]))
donor_pool = [(e["clip_id"], e["gold_coc"]) for e in events]
sel = events[args.shard_idx::args.num_shards]
if args.limit:
    sel = sel[:args.limit]
arms = [a for a in args.arms.split(",") if a]
print(f"[dmap] {len(sel)} events, arms={arms}, steps={args.steps}, "
      f"dtype={'bf16' if args.bf16 else 'fp32'}", flush=True)

# ---------------- model ----------------
# MIXED PRECISION, deliberately.
#
# `torch_dtype` is NOT honoured here: config.model_dtype=bfloat16 with keep_same_dtype=True makes
# the constructor cast expert/action_in_proj/action_out_proj to bf16 regardless, while ~half the
# VLM stays fp32. That mixture raises "mat1 and mat2 must have the same dtype" and is what killed
# the ODE fp32 arm. Casting EVERYTHING to fp32 instead OOMs at 79 GB (the fp32 vision tower plus
# an image-heavy prefix cache).
#
# So: VLM in bf16 -- it only produces the KV cache, which is exactly what the deployed model and
# the surrogate both do -- and the EXPERT in fp32, because that is where the Jacobian and the
# log-determinant accumulate. The cache is cast to fp32 at the boundary. Note this makes the
# measured map MORE precise than the deployed bf16 one; that is intended, since bf16 rounding is
# not differentiable and "the density of the bf16 map" is not a well-defined object. The object
# of study is the 10-step Euler map with the velocity field evaluated exactly.
DT = torch.bfloat16 if args.bf16 else torch.float32
# Device/precision split. On ONE 80 GB A100 the fp32 VLM cannot coexist with the expert: weights
# are only ~41 GB but the fp32 multi-camera vision prefill adds ~37 GB and it OOMs at 79 GB. The
# VLM and the expert run strictly sequentially (one prefill, then 640 expert passes), so giving
# them a GPU each costs nothing but a GPU -- and it removes the bf16 KV cache, which was the last
# precision caveat on the donor contrast (each contrast differences two logdet ~ -240 terms, so
# bf16 relative error is ~0.2 nats: negligible vs wrongtraj, NOT vs donor).
VDEV, EDEV = ("cuda:0", "cuda:1") if args.device_split else ("cuda:0", "cuda:0")
VDT = torch.float32 if (args.device_split and not args.bf16) else torch.bfloat16
model = TrainableAlpamayoR1.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
model.vlm.to(device=VDEV, dtype=VDT)
for mod in (model.expert, model.action_in_proj, model.action_out_proj):
    mod.to(device=EDEV, dtype=DT)
model.action_space.to(device=EDEV)
_edt = {str(q.dtype) for q in model.expert.parameters()}
_vdt = {str(q.dtype) for q in model.vlm.parameters()}
assert _edt == {str(DT)}, f"mixed expert dtypes: {_edt}"
assert _vdt == {str(VDT)}, f"mixed vlm dtypes: {_vdt}"
print(f"[dmap] vlm={VDT} on {VDEV} ({torch.cuda.memory_allocated(VDEV)/2**30:.1f} GiB)  "
      f"expert={DT} on {EDEV} ({torch.cuda.memory_allocated(EDEV)/2**30:.1f} GiB)", flush=True)
adims = tuple(model.action_space.get_action_space_dims())
D = int(np.prod(adims))
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
print(f"[dmap] D={D} adims={adims}", flush=True)


def kv_for(data, coc_text, blank_vision=False):
    s = {"image_frames": (torch.zeros_like(data["image_frames"]) if blank_vision
                          else data["image_frames"]),
         "camera_indices": data["camera_indices"],
         "relative_timestamps": data["relative_timestamps"],
         "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
         "ego_history_rot": data["ego_history_rot"].squeeze(0),
         "cot": coc_text}
    s["tokenized_data"] = pre(s)
    batch = collate_fn_from_model_config([dict(s)], model_config=model.config,
        include_camera_ids=True, include_frame_nums=True, chat_template_version="r1")
    td = to_device({"td": dict(batch["tokenized_data"])}, VDEV)["td"]
    ids = td.pop("input_ids")
    ids = model.fuse_traj_tokens(ids, {k: data[k].to(VDEV) for k in
        ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")})
    with torch.no_grad():
        out = model.vlm.model(input_ids=ids, use_cache=True, **td)
    fs = (ids == fstart).nonzero(as_tuple=False)
    assert fs.shape[0] == 1, f"expected one <traj_future_start>, got {fs.shape[0]}"
    cache = out.past_key_values
    plen = int(fs[0, 1].item()) + 1
    cache.crop(plen)
    delta_out = int(out.rope_deltas.flatten()[0].item()) + int(cache.get_seq_length())
    # cast at the VLM -> expert boundary (see the mixed-precision note above)
    snap = [(l.keys.to(device=EDEV, dtype=DT), l.values.to(device=EDEV, dtype=DT))
            for l in cache.layers]
    del cache, out
    return snap, delta_out, plen


def make_vfield(snap, delta):
    def vfield(x, t):
        B = x.shape[0]
        cache = DynamicCache(ddp_cache_data=[
            (k.expand(B, *k.shape[1:]), v.expand(B, *v.shape[1:])) for k, v in snap])
        tt = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(1).expand(B)
        tt = tt.reshape(B, *([1] * (x.dim() - 1)))
        emb = model.action_in_proj(x, tt)
        pos = torch.arange(emb.shape[1], device=emb.device).view(1, 1, -1).expand(3, B, -1).clone() + delta
        fwd = {"is_causal": False} if model.config.expert_non_causal_attention else {}
        eo = model.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache,
                          attention_mask=None, use_cache=False, **fwd)
        return model.action_out_proj(eo.last_hidden_state[:, -emb.shape[1]:]).view(B, *adims)
    return vfield


def gold_action(data):
    return model.action_space.traj_to_action(
        traj_history_xyz=data["ego_history_xyz"].to(EDEV), traj_history_rot=data["ego_history_rot"].to(EDEV),
        traj_future_xyz=data["ego_future_xyz"].to(EDEV), traj_future_rot=data["ego_future_rot"].to(EDEV)
    ).reshape(1, *adims).to(device=EDEV, dtype=DT)


def _rng(tag, cid, ei):
    key = f"{args.donor_seed}|{tag}|{cid}|{int(ei)}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def _future_xy(d):
    return np.asarray(d["ego_future_xyz"].detach().cpu()).reshape(-1, 3)[:, :2]


def _ade(a, b):
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


rows, pool, t0run = [], [], time.time()
for i, ev in enumerate(sel):
    try:
        data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
        a_star = gold_action(data)

        conds = [("gold", ev["gold_coc"], dict())]
        if "donor" in arms:
            r = _rng("random", ev["clip_id"], ev["event_idx"])
            cands = [j for j, (cid, _) in enumerate(donor_pool) if cid != ev["clip_id"]]
            conds.append(("donor", donor_pool[cands[int(r.integers(len(cands)))]][1], dict()))
        if "blankvision" in arms:
            conds.append(("blankvision", ev["gold_coc"], dict(blank=True)))
        if "wrongtraj" in arms and pool:
            r = _rng("wrongtraj", ev["clip_id"], ev["event_idx"])
            mine = _future_xy(data)
            best_k, best_a = None, -1.0
            for _try in range(max(1, args.donor_traj_max_tries)):
                k = int(r.integers(len(pool)))
                a = _ade(mine, pool[k][1])
                if a > best_a:
                    best_k, best_a = k, a
                if a >= args.donor_traj_min_ade:
                    break
            conds.append(("wrongtraj", ev["gold_coc"], dict(x=pool[best_k][0], ade=best_a)))

        for name, txt, opt in conds:
            snap, delta, plen = kv_for(data, txt, blank_vision=opt.get("blank", False))
            vf = make_vfield(snap, delta)
            y = opt.get("x", a_star)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16
                   else contextlib.nullcontext())
            t0 = time.time()
            with ctx, sdpa_kernel([SDPBackend.MATH]):
                lp, diag = logp_discrete_map(y, vf, args.steps, D, tchunk=args.tchunk)
            sv = diag.pop("svals")
            rows.append(dict(clip_id=ev["clip_id"], event_idx=ev["event_idx"],
                             event_cluster=ev["event_cluster"], cond=name, logp=lp,
                             prefix_len=plen, secs=time.time() - t0,
                             donor_ade=float(opt.get("ade", np.nan)),
                             svals=sv.astype(np.float64), **diag))
            del snap, vf
            torch.cuda.empty_cache()

        if len(pool) >= 64:
            pool.pop(0)
        pool.append((a_star.detach().clone(), _future_xy(data)))

        g = next(r for r in rows[-len(conds):] if r["cond"] == "gold")
        msg = " ".join(f"{r['cond']}={r['logp']-g['logp']:+.4f}" for r in rows[-len(conds):]
                       if r["cond"] != "gold")
        print(f"[dmap] {i+1}/{len(sel)} {str(ev['clip_id'])[:8]} logp_gold={g['logp']:+.3f} "
              f"rt={g['roundtrip']:.1e} rank(1e-6)={g['rank_1em6']}/{D} cond={g['cond_G']:.1e} "
              f"| gold-minus: {msg} | {time.time()-t0run:.0f}s", flush=True)
    except Exception as e:
        print(f"[dmap] SKIP {ev['clip_id']} idx={ev['event_idx']}: {type(e).__name__}: {e}", flush=True)
        continue

    if (i + 1) % 10 == 0:
        pd.DataFrame(rows).to_parquet(args.out)

pd.DataFrame(rows).to_parquet(args.out)
print(f"[dmap] wrote {args.out} rows={len(rows)}", flush=True)
