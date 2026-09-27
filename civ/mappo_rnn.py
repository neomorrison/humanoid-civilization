"""Multi-agent PPO with memory: every person carries a GRU state through their life.

The actor is  obs -> dense -> GRU -> dense -> action logits.  A person's hidden state
starts at zero when they are born (or when their slot is taken by someone new) and is
carried from hour to hour for the rest of their life, so they can remember what happened
earlier in the day, notice patterns and adapt to changes they were never trained on.
The critic stays feed-forward on the centralised observation.

Training uses rollouts of `horizon` hours per person; the loss re-runs the GRU over each
rollout from the hidden state it started with (truncated backpropagation through time),
minibatching over people rather than over hours.
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

from .ppo import RunningNorm, gae, init_mlp, mlp


@dataclass
class RNNConfig:
    enc: int = 256
    hidden: int = 128
    head: int = 128
    critic: tuple = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    epochs: int = 4
    minibatches: int = 4
    horizon: int = 64
    max_grad_norm: float = 0.5


def init_params(key, obs_dim, cobs_dim, n_actions, cfg: RNNConfig):
    ke, kg, kh, kc = jax.random.split(key, 4)
    H, E = cfg.hidden, cfg.enc
    orth = jax.nn.initializers.orthogonal
    gru = dict(Wx=orth(1.0)(kg, (E, 3 * H), jnp.float32), Wh=orth(1.0)(jax.random.fold_in(kg, 1), (H, 3 * H), jnp.float32),
               b=jnp.zeros(3 * H, jnp.float32))
    actor = dict(enc=init_mlp(ke, [obs_dim, E]), gru=gru, head=init_mlp(kh, [H, cfg.head, n_actions], out_scale=0.01))
    return dict(actor=actor, critic=init_mlp(kc, [cobs_dim, *cfg.critic, 1], out_scale=1.0))


def gru_cell(p, x, h):
    H = h.shape[-1]
    gx = x @ p["Wx"] + p["b"]
    gh = h @ p["Wh"]
    z = jax.nn.sigmoid(gx[..., :H] + gh[..., :H])
    r = jax.nn.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
    n = jnp.tanh(gx[..., 2 * H:] + r * gh[..., 2 * H:])
    return (1 - z) * n + z * h


def actor_step(pa, obs, h):
    x = jax.nn.elu(mlp(pa["enc"], obs))
    h = gru_cell(pa["gru"], x, h)
    return mlp(pa["head"], h), h


@jax.jit
def act(params, obs, cobs, h, reset, key):
    h = h * (1.0 - reset[:, None])
    logits, h = actor_step(params["actor"], obs, h)
    a = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)
    v = mlp(params["critic"], cobs)[:, 0]
    return a, jnp.take_along_axis(logp, a[:, None], 1)[:, 0], v, logits, h


@jax.jit
def value(params, cobs):
    return mlp(params["critic"], cobs)[:, 0]


def make_update(cfg: RNNConfig):
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))

    def loss_fn(params, b):
        # b: sequences over time for a group of people: obs (T, n, d), reset (T, n), h0 (n, H), ...
        def step(h, inp):
            o, rs = inp
            h = h * (1.0 - rs[:, None])
            lg, h = actor_step(params["actor"], o, h)
            return h, lg
        _, logits = jax.lax.scan(step, b["h0"], (b["obs"], b["reset"]))
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, b["act"][..., None], -1)[..., 0]
        ratio = jnp.exp(logp - b["logp"])
        adv = b["adv"]
        w, wd = b["mask"], b["dmask"]
        wm = lambda x: (x * w).sum() / jnp.maximum(w.sum(), 1.0)
        dm = lambda x: (x * wd).sum() / jnp.maximum(wd.sum(), 1.0)
        pg = -dm(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))
        v = mlp(params["critic"], b["cobs"])[..., 0]
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
    pa = jax.tree_util.tree_map(np.asarray, params["actor"])
    arrays = dict(enc_W=pa["enc"][0][0], enc_b=pa["enc"][0][1], gru_Wx=pa["gru"]["Wx"], gru_Wh=pa["gru"]["Wh"],
                  gru_b=pa["gru"]["b"], n_head=np.int32(len(pa["head"])))
    for i, (W, b) in enumerate(pa["head"]):
        arrays[f"head_W{i}"], arrays[f"head_b{i}"] = W, b
    arrays["obs_mean"] = norm.mean.astype(np.float32)
    arrays["obs_std"] = np.sqrt(norm.var + 1e-8).astype(np.float32)
    arrays["obs_clip"] = np.float32(norm.clip)
    arrays["meta_json"] = np.frombuffer(json.dumps(dict(meta, recurrent=True)).encode(), dtype=np.uint8)
    np.savez(path, **arrays)


class NumpyRNNPolicy:
    """Framework-free recurrent actor; keeps one hidden state per person slot."""

    def __init__(self, path):
        z = np.load(path)
        self.enc_W, self.enc_b = z["enc_W"], z["enc_b"]
        self.Wx, self.Wh, self.b = z["gru_Wx"], z["gru_Wh"], z["gru_b"]
        self.head = [(z[f"head_W{i}"], z[f"head_b{i}"]) for i in range(int(z["n_head"]))]
        self.mean, self.std, self.clip = z["obs_mean"], z["obs_std"], float(z["obs_clip"])
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        self.h = None

    def reset(self, mask):
        if self.h is not None:
            self.h[np.asarray(mask, bool)] = 0

    def logits(self, obs):
        x = np.clip((obs - self.mean) / self.std, -self.clip, self.clip)
        x = x @ self.enc_W + self.enc_b
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        if self.h is None or self.h.shape[0] != len(obs):
            self.h = np.zeros((len(obs), self.Wh.shape[0]), np.float32)
        H = self.h.shape[1]
        gx, gh = x @ self.Wx + self.b, self.h @ self.Wh
        sig = lambda v: 1 / (1 + np.exp(-v))
        zg, rg = sig(gx[:, :H] + gh[:, :H]), sig(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gx[:, 2 * H:] + rg * gh[:, 2 * H:])
        self.h = ((1 - zg) * n + zg * self.h).astype(np.float32)
        y = self.h
        for i, (W, b) in enumerate(self.head):
            y = y @ W + b
            if i < len(self.head) - 1:
                y = np.where(y > 0, y, np.expm1(np.minimum(y, 0)))
        return y

    def sample(self, obs, rng):
        lg = self.logits(obs)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        u = rng.random((len(p), 1))
        return np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)   # float rounding can leave the cdf just below u


def train(env, out, minutes, cfg: RNNConfig | None = None, seed=0, every=50, on_snapshot=None, resume=None,
          max_iters=None):
    cfg = cfg or RNNConfig()
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params = init_params(k0, env.obs_dim, env.cobs_dim, env.act_dim, cfg)
    opt, update = make_update(cfg)
    opt_state = opt.init(params)
    onorm, cnorm = RunningNorm(env.obs_dim), RunningNorm(env.cobs_dim)
    it = 0
    T, N, H = cfg.horizon, env.N, cfg.hidden
    h = np.zeros((N, H), np.float32)
    if resume:
        with open(resume, "rb") as f:
            ck = pickle.load(f)
        params, it = jax.tree_util.tree_map(jnp.asarray, ck["params"]), ck["iteration"]
        opt_state = opt.init(params)
        onorm.load(ck["onorm"])
        cnorm.load(ck["cnorm"])
    obs, cobs = env.observe_all()
    onorm.update(obs)
    cnorm.update(cobs)
    buf = dict(obs=np.zeros((T, N, env.obs_dim), np.float32), cobs=np.zeros((T, N, env.cobs_dim), np.float32),
               act=np.zeros((T, N), np.int32), logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32),
               rew=np.zeros((T, N), np.float32), done=np.zeros((T, N), np.float32),
               logits=np.zeros((T, N, env.act_dim), np.float32), mask=np.ones((T, N), np.float32),
               dmask=np.ones((T, N), np.float32), reset=np.zeros((T, N), np.float32))
    log_path = os.path.join(out, "train_log.jsonl")
    t0 = time.time()
    samples = 0

    def snapshot():
        meta = dict(iteration=it, samples=samples, obs_dim=env.obs_dim)
        path = os.path.join(out, "snapshots", f"policy_it{it:05d}.npz")
        export(path, params, onorm, meta)
        export(os.path.join(out, "policy.npz"), params, onorm, meta)
        with open(os.path.join(out, "checkpoint.pkl"), "wb") as f:
            pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), iteration=it, onorm=onorm.state(),
                             cnorm=cnorm.state(), recurrent=True, obs_names=getattr(env, "obs_names", None),
                             cobs_names=getattr(env, "cobs_names", None), act_names=getattr(env, "act_names", None)), f)
        if on_snapshot:
            on_snapshot(path, it, samples)

    snapshot()
    start_it = it
    while (time.time() - t0) / 60 < minutes and (max_iters is None or it - start_it < max_iters):
        t_it = time.time()
        rsum = 0.0
        h0 = h.copy()
        for t in range(T):
            reset = env.reset_mask().astype(np.float32)
            on, cn = onorm(obs), cnorm(cobs)
            key, ka = jax.random.split(key)
            a, logp, v, lg, hn = act(params, on, cn, jnp.asarray(h), jnp.asarray(reset), ka)
            a = np.asarray(a)
            h = np.asarray(hn)
            buf["obs"][t], buf["cobs"][t], buf["act"][t], buf["reset"][t] = on, cn, a, reset
            buf["logp"][t], buf["val"][t], buf["logits"][t] = np.asarray(logp), np.asarray(v), np.asarray(lg)
            buf["mask"][t] = env.alive_mask()
            buf["dmask"][t] = env.decision_mask()
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
        seq = dict(buf, adv=adv, ret=ret)
        mb = N // cfg.minibatches
        stats = []
        for _ in range(cfg.epochs):
            perm = np.random.permutation(N)
            for j in range(cfg.minibatches):
                ix = perm[j * mb:(j + 1) * mb]
                b = {k: seq[k][:, ix] for k in ("obs", "cobs", "act", "logp", "adv", "ret", "logits", "mask", "dmask", "reset")}
                b["h0"] = h0[ix]
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
