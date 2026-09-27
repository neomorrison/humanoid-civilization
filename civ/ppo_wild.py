"""PPO for Wild, entirely inside JAX: act, step the worlds, compute advantages and update in one jitted loop.

One network is shared by everyone; each person's inner values (genes) are part of what it sees,
so the same network serves different temperaments. Worlds whose people all die restart.

    python -m civ.ppo_wild --minutes 60 --out runs/wild
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import wild
from .ppo import init_mlp, mlp


class PPOConfig(NamedTuple):
    worlds: int = 64
    horizon: int = 32
    hidden: tuple = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vcoef: float = 0.5
    ent: float = 0.01
    epochs: int = 3
    minibatches: int = 4
    max_grad_norm: float = 0.5


def init_params(key, obs_dim, n_act, pc: PPOConfig):
    ka, kc = jax.random.split(key)
    return dict(actor=init_mlp(ka, [obs_dim, *pc.hidden, n_act], out_scale=0.01),
                critic=init_mlp(kc, [obs_dim, *pc.hidden, 1], out_scale=1.0))


def make_train(cfg: wild.Config, pc: PPOConfig):
    M, S, T = pc.worlds, cfg.S, pc.horizon
    n_act = wild.n_actions(cfg)
    opt = optax.chain(optax.clip_by_global_norm(pc.max_grad_norm), optax.adam(pc.lr))
    v_reset = jax.vmap(lambda k: wild.reset(k, cfg))
    v_step = jax.vmap(lambda s, a, k: wild.step(s, a, k, cfg))
    v_obs = jax.vmap(lambda s: wild.observe(s, cfg))

    def env_step(carry, _):
        params, st, obs, key = carry
        key, ka, ks, kr = jax.random.split(key, 4)
        flat = obs.reshape(M * S, -1)
        logits = mlp(params["actor"], flat)
        a = jax.random.categorical(ka, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], 1)[:, 0]
        v = mlp(params["critic"], flat)[:, 0]
        alive = st.alive.reshape(-1)
        st2, r, info = v_step(st, a.reshape(M, S), jax.random.split(ks, M))
        # a world with nobody left starts over
        extinct = ~st2.alive.any(-1)
        fresh = v_reset(jax.random.split(kr, M))
        st2 = jax.tree_util.tree_map(lambda f, s: jnp.where(extinct.reshape((M,) + (1,) * (s.ndim - 1)), f, s), fresh, st2)
        # a slot ends when its person dies (or the world restarts); a newborn in the slot starts a new life
        done = (info["died"] | extinct[:, None]).reshape(-1)
        obs2 = v_obs(st2)
        out = dict(obs=flat.astype(jnp.bfloat16), act=a, logp=logp, val=v, rew=r.reshape(-1), done=done * 1.0,
                   mask=alive * 1.0)
        stats = dict(alive=st.alive.sum(-1).mean(), births=info["born"].sum(), deaths=info["died"].sum(),
                     fires=info["fires"].sum(), shelters=info["shelters"].sum(), gives=info["gives"].sum(),
                     meals=info["meals"].sum(), drinks=info["drinks"].sum(), grabs=info["grabs"].sum(),
                     mates=info["mates"].sum(), extinct=extinct.sum(), reward=(r * st.alive).sum() / jnp.maximum(st.alive.sum(), 1))
        return (params, st2, obs2, key), (out, stats)

    def gae(rew, val, done, last):
        def f(carry, x):
            adv_next, v_next = carry
            r, v, d = x
            delta = r + pc.gamma * v_next * (1 - d) - v
            adv = delta + pc.gamma * pc.lam * (1 - d) * adv_next
            return (adv, v), adv
        _, adv = jax.lax.scan(f, (jnp.zeros_like(last), last), (rew, val, done), reverse=True)
        return adv, adv + val

    def loss_fn(params, b):
        logits = mlp(params["actor"], b["obs"].astype(jnp.float32))
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, b["act"][:, None], 1)[:, 0]
        ratio = jnp.exp(lp - b["logp"])
        w = b["mask"]
        wm = lambda x: (x * w).sum() / jnp.maximum(w.sum(), 1.0)
        pg = -wm(jnp.minimum(ratio * b["adv"], jnp.clip(ratio, 1 - pc.clip, 1 + pc.clip) * b["adv"]))
        v = mlp(params["critic"], b["obs"].astype(jnp.float32))[:, 0]
        vl = wm((v - b["ret"]) ** 2)
        ent = wm(-(jnp.exp(lp_all) * lp_all).sum(-1))
        return pg + pc.vcoef * vl - pc.ent * ent, dict(pg=pg, vl=vl, ent=ent)

    @jax.jit
    def iteration(params, opt_state, st, obs, key):
        (params, st, obs, key), (buf, stats) = jax.lax.scan(env_step, (params, st, obs, key), None, T)
        last = mlp(params["critic"], obs.reshape(M * S, -1))[:, 0]
        adv, ret = gae(buf["rew"], buf["val"], buf["done"], last)
        w = buf["mask"]
        mu = (adv * w).sum() / jnp.maximum(w.sum(), 1)
        sd = jnp.sqrt((((adv - mu) ** 2) * w).sum() / jnp.maximum(w.sum(), 1)) + 1e-8
        data = dict(obs=buf["obs"], act=buf["act"], logp=buf["logp"], adv=(adv - mu) / sd, ret=ret, mask=w)
        data = jax.tree_util.tree_map(lambda x: x.reshape((T * M * S,) + x.shape[2:]), data)
        n = T * M * S
        mb = n // pc.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, n)

            def minibatch(carry, j):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (j * mb,), (mb,))
                b = jax.tree_util.tree_map(lambda x: x[idx], data)
                (_, st_), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
                u, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, u), opt_state), st_
            (params, opt_state), mstats = jax.lax.scan(minibatch, (params, opt_state), jnp.arange(pc.minibatches))
            return (params, opt_state), mstats
        key, ke = jax.random.split(key)
        (params, opt_state), lstats = jax.lax.scan(epoch, (params, opt_state), jax.random.split(ke, pc.epochs))
        summary = jax.tree_util.tree_map(lambda x: x.mean(), stats)
        totals = {k: stats[k].sum() for k in ("births", "deaths", "fires", "shelters", "gives", "meals", "drinks", "grabs",
                                               "mates", "extinct")}
        return params, opt_state, st, obs, key, dict(summary, **{f"n_{k}": v for k, v in totals.items()},
                                                    ent=lstats["ent"].mean(), vl=lstats["vl"].mean())

    def init(key):
        k0, k1 = jax.random.split(key)
        st = v_reset(jax.random.split(k0, M))
        obs = v_obs(st)
        params = init_params(k1, obs.shape[-1], n_act, pc)
        return params, opt.init(params), st, obs

    return init, iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default="runs/wild")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg, pc = wild.Config(), PPOConfig(worlds=args.worlds)
    os.makedirs(args.out, exist_ok=True)
    init, iteration = make_train(cfg, pc)
    key = jax.random.PRNGKey(args.seed)
    key, ki = jax.random.split(key)
    params, opt_state, st, obs = init(ki)
    t0 = time.time()
    it = 0
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    while (time.time() - t0) / 60 < args.minutes:
        t1 = time.time()
        params, opt_state, st, obs, key, stats = iteration(params, opt_state, st, obs, key)
        stats = {k: float(v) for k, v in stats.items()}
        it += 1
        sps = pc.horizon * pc.worlds * cfg.S / (time.time() - t1)
        rec = dict(it=it, minutes=(time.time() - t0) / 60, samples=it * pc.horizon * pc.worlds * cfg.S, sps=sps, **stats)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 20 == 0:
            print(f"it={it} t={rec['minutes']:.1f}m sps={sps:,.0f} alive={stats['alive']:.1f} r={stats['reward']:+.4f} "
                  f"ent={stats['ent']:.2f} drinks={stats['n_drinks']:.0f} meals={stats['n_meals']:.0f} "
                  f"grabs={stats['n_grabs']:.0f} fires={stats['n_fires']:.0f} shelters={stats['n_shelters']:.0f} "
                  f"gives={stats['n_gives']:.0f} births={stats['n_births']:.0f} deaths={stats['n_deaths']:.0f} "
                  f"extinct={stats['n_extinct']:.0f}", flush=True)
        if it % 200 == 0:
            with open(os.path.join(args.out, "checkpoint.pkl"), "wb") as f:
                pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), iteration=it,
                                 config=cfg._asdict(), ppo=pc._asdict()), f)
    with open(os.path.join(args.out, "checkpoint.pkl"), "wb") as f:
        pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), iteration=it, config=cfg._asdict(),
                         ppo=pc._asdict()), f)


if __name__ == "__main__":
    main()
