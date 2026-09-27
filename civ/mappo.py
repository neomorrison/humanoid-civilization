"""Multi-agent PPO with discrete actions (JAX).

One actor network is shared by every agent (its role is an input), so a
worker, the farmer and the police officer are the same "species" with
different jobs. The critic sees the whole economy (centralised training,
decentralised execution): each agent's policy only uses its own observation.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .numpy_policy import NumpyPolicy  # noqa: F401  (the framework-free actor, re-exported)
from .ppo import RunningNorm, gae, init_mlp, mlp


@dataclass
class MAPPOConfig:
    hidden: tuple = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatches: int = 4
    horizon: int = 64
    max_grad_norm: float = 0.5


def init_params(key, obs_dim, cobs_dim, n_actions, cfg):
    ka, kc = jax.random.split(key)
    return dict(actor=init_mlp(ka, [obs_dim, *cfg.hidden, n_actions], out_scale=0.01),
                critic=init_mlp(kc, [cobs_dim, *cfg.hidden, 1], out_scale=1.0))


@jax.jit
def act(params, obs, cobs, key):
    logits = mlp(params["actor"], obs)
    a = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)
    return a, jnp.take_along_axis(logp, a[:, None], 1)[:, 0], mlp(params["critic"], cobs)[:, 0], logits


@jax.jit
def value(params, cobs):
    return mlp(params["critic"], cobs)[:, 0]


def make_update(cfg):
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))

    def loss_fn(params, b):
        logits = mlp(params["actor"], b["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, b["act"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - b["logp"])
        adv = b["adv"]
        w, wd = b["mask"], b["dmask"]
        wm = lambda x: (x * w).sum() / jnp.maximum(w.sum(), 1.0)      # mean over living agents only
        dm = lambda x: (x * wd).sum() / jnp.maximum(wd.sum(), 1.0)    # mean over steps where they chose
        pg = -dm(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))
        v = mlp(params["critic"], b["cobs"])[:, 0]
        vl = wm((v - b["ret"]) ** 2)
        p = jnp.exp(logp_all)
        ent = dm(-(p * logp_all).sum(-1))
        old = jax.nn.log_softmax(b["logits"])
        kl = dm((jnp.exp(old) * (old - logp_all)).sum(-1))
        return pg + cfg.value_coef * vl - cfg.entropy_coef * ent, dict(pg=pg, vl=vl, ent=ent, kl=kl)

    @jax.jit
    def update(params, opt_state, b):
        (_, st), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, st

    return opt, update


def export(path, params, norm, meta):
    arrays = {}
    for i, (W, b) in enumerate(params["actor"]):
        arrays[f"W{i}"] = np.asarray(W, np.float32)
        arrays[f"b{i}"] = np.asarray(b, np.float32)
    arrays["obs_mean"] = norm.mean.astype(np.float32)
    arrays["obs_std"] = np.sqrt(norm.var + 1e-8).astype(np.float32)
    arrays["obs_clip"] = np.float32(norm.clip)
    arrays["n_layers"] = np.int32(len(params["actor"]))
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **arrays)


def train(env, out, minutes, cfg: MAPPOConfig | None = None, seed=0, every=50, on_snapshot=None, resume=None,
          init_from=None, names_if_missing=None, max_iters=None):
    cfg = cfg or MAPPOConfig()
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params = init_params(k0, env.obs_dim, env.cobs_dim, env.act_dim, cfg)
    opt, update = make_update(cfg)
    opt_state = opt.init(params)
    onorm, cnorm = RunningNorm(env.obs_dim), RunningNorm(env.cobs_dim)
    it = 0
    if resume:
        with open(resume, "rb") as f:
            ck = pickle.load(f)
        params, it = ck["params"], ck["iteration"]
        opt_state = opt.init(params)
        onorm.load(ck["onorm"])
        cnorm.load(ck["cnorm"])
    elif init_from:
        # warm start: carry the weights of a policy trained on a different layout, matched by name
        from .warmstart import transplant
        with open(init_from, "rb") as f:
            ck = pickle.load(f)
        for k, v in (names_if_missing or {}).items():
            ck.setdefault(k, v)
        new, on_s, cn_s, rep = transplant(ck, jax.tree_util.tree_map(np.asarray, params),
                                          env.obs_names, env.cobs_names, env.act_names)
        params = jax.tree_util.tree_map(jnp.asarray, new)
        opt_state = opt.init(params)
        onorm.load(on_s)
        cnorm.load(cn_s)
        it = ck["iteration"]
        print(f"warm start from {init_from} (it {it}): {len(rep['new_inputs'])} new inputs, "
              f"{len(rep['dropped_inputs'])} dropped, new actions {rep['new_actions']}, dropped {rep['dropped_actions']}",
              flush=True)
    obs, cobs = env.observe_all()
    onorm.update(obs)
    cnorm.update(cobs)
    T, N = cfg.horizon, env.N
    buf = dict(obs=np.zeros((T, N, env.obs_dim), np.float32), cobs=np.zeros((T, N, env.cobs_dim), np.float32),
               act=np.zeros((T, N), np.int32), logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32),
               rew=np.zeros((T, N), np.float32), done=np.zeros((T, N), np.float32),
               logits=np.zeros((T, N, env.act_dim), np.float32), mask=np.ones((T, N), np.float32),
               dmask=np.ones((T, N), np.float32))
    alive_mask = getattr(env, "alive_mask", None)
    decision_mask = getattr(env, "decision_mask", None)     # steps where the agent actually chose its action
    log_path = os.path.join(out, "train_log.jsonl")
    t0 = time.time()
    samples = 0

    def snapshot():
        path = os.path.join(out, "snapshots", f"policy_it{it:05d}.npz")
        export(path, params, onorm, dict(iteration=it, samples=samples, obs_dim=env.obs_dim))
        export(os.path.join(out, "policy.npz"), params, onorm, dict(iteration=it, samples=samples, obs_dim=env.obs_dim))
        with open(os.path.join(out, "checkpoint.pkl"), "wb") as f:
            pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), iteration=it,
                             onorm=onorm.state(), cnorm=cnorm.state(), obs_names=getattr(env, "obs_names", None),
                             cobs_names=getattr(env, "cobs_names", None), act_names=getattr(env, "act_names", None)), f)
        if on_snapshot:
            on_snapshot(path, it, samples)

    snapshot()
    start_it = it
    while (time.time() - t0) / 60 < minutes and (max_iters is None or it - start_it < max_iters):
        t_it = time.time()
        rsum = 0.0
        for t in range(T):
            on, cn = onorm(obs), cnorm(cobs)
            key, ka = jax.random.split(key)
            a, logp, v, lg = act(params, on, cn, ka)
            a = np.asarray(a)
            buf["obs"][t], buf["cobs"][t], buf["act"][t] = on, cn, a
            buf["logp"][t], buf["val"][t], buf["logits"][t] = np.asarray(logp), np.asarray(v), np.asarray(lg)
            if alive_mask is not None:
                buf["mask"][t] = alive_mask()
            buf["dmask"][t] = decision_mask() if decision_mask is not None else buf["mask"][t]
            obs, cobs, rew, done, info = env.step(a)
            rew = rew + cfg.gamma * buf["val"][t] * info["timeout"]
            buf["rew"][t], buf["done"][t] = rew, done
            rsum += float((rew * buf["mask"][t]).sum() / max(buf["mask"][t].sum(), 1))
            onorm.update(obs)
            cnorm.update(cobs)
        samples += T * N
        last = np.asarray(value(params, cnorm(cobs)))
        adv, ret = gae(buf["rew"], buf["val"], buf["done"], last, cfg.gamma, cfg.lam)
        live = buf["dmask"] > 0
        adv = (adv - adv[live].mean()) / (adv[live].std() + 1e-8)
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in buf.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        mb = T * N // cfg.minibatches
        stats = []
        for _ in range(cfg.epochs):
            perm = np.random.permutation(T * N)
            for j in range(cfg.minibatches):
                ix = perm[j * mb:(j + 1) * mb]
                b = {k: flat[k][ix] for k in ["obs", "cobs", "act", "logp", "adv", "ret", "logits", "mask", "dmask"]}
                params, opt_state, st = update(params, opt_state, b)
                stats.append({k: float(v) for k, v in st.items()})
        it += 1
        recent = env.stats[-64:]
        econ = {k: float(np.mean([s[k] for s in recent])) for k in recent[0]
                if not isinstance(recent[0][k], list)} if recent else {}
        rec = dict(it=it, minutes=(time.time() - t0) / 60, samples=samples, sps=T * N / (time.time() - t_it),
                   reward=rsum / T, entropy=float(np.mean([s["ent"] for s in stats])),
                   kl=float(np.mean([s["kl"] for s in stats])), econ=econ)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 10 == 0:
            e = {k: v for k, v in econ.items() if isinstance(v, float)}
            show = " ".join(f"{k}={v:.1f}" for k, v in list(e.items())[:12])
            print(f"it={it} t={rec['minutes']:.1f}m sps={rec['sps']:.0f} r={rec['reward']:+.4f} "
                  f"ent={rec['entropy']:.2f} {show}", flush=True)
        if it % every == 0:
            snapshot()
    snapshot()
    return params
