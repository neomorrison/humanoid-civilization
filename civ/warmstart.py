"""Carry what a policy has learned into a changed environment.

Checkpoints store the names of their observation inputs and actions. Warm-starting
builds the new network, then copies weights by name:

* an input the old network also had keeps its weights and normalisation statistics;
  a new input starts with zero weights, so it changes nothing until it is learned;
* an action the old network also had keeps its output weights; a new action starts
  with zero weights and a bias just below a typical action's, so it gets tried without disturbing
  the routines already learned;
* hidden layers are copied as they are (their sizes must match).
"""
from __future__ import annotations

import numpy as np

NEW_ACTION_BIAS = -1.0      # relative to the median existing action bias


def _rows(W_old, old_names, W_new, new_names):
    W = np.zeros_like(np.asarray(W_new))
    idx = {n: i for i, n in enumerate(old_names)}
    for j, n in enumerate(new_names):
        if n in idx:
            W[j] = W_old[idx[n]]
    return W


def _norm(state, old_names, new_names):
    idx = {n: i for i, n in enumerate(old_names)}
    mean = np.zeros(len(new_names))
    var = np.ones(len(new_names))
    for j, n in enumerate(new_names):
        if n in idx:
            mean[j], var[j] = state["mean"][idx[n]], state["var"][idx[n]]
    return dict(mean=mean, var=var, count=state["count"], clip=state["clip"])


def transplant(ck: dict, fresh: dict, obs_names, cobs_names, act_names):
    """Return (params, onorm_state, cnorm_state, report) for the new layout.

    ck: an old checkpoint dict (params, onorm, cnorm, obs_names, cobs_names, act_names).
    fresh: freshly initialised params for the new layout (same hidden sizes).
    """
    old_obs, old_cobs, old_act = ck["obs_names"], ck["cobs_names"], ck["act_names"]
    out = {}
    for net, olds, news in (("actor", old_obs, obs_names), ("critic", old_cobs, cobs_names)):
        old_layers, new_layers = ck["params"][net], fresh[net]
        if len(old_layers) != len(new_layers):
            raise ValueError(f"{net}: layer count differs ({len(old_layers)} vs {len(new_layers)})")
        layers = []
        for i, ((Wo, bo), (Wn, bn)) in enumerate(zip(old_layers, new_layers)):
            Wo, bo = np.asarray(Wo), np.asarray(bo)
            if i == 0:
                W = _rows(Wo, olds, Wn, news)
            else:
                if Wo.shape[0] != np.asarray(Wn).shape[0]:
                    raise ValueError(f"{net} layer {i}: hidden sizes differ")
                W = Wo
            b = bo
            if i == len(old_layers) - 1 and net == "actor":
                idx = {n: k for k, n in enumerate(old_act)}
                W = np.zeros((Wo.shape[0], len(act_names)), np.float32)
                b = np.full(len(act_names), np.median(bo) + NEW_ACTION_BIAS, np.float32)
                for k, n in enumerate(act_names):
                    if n in idx:
                        W[:, k], b[k] = Wo[:, idx[n]], bo[idx[n]]
            layers.append((np.asarray(W, np.float32), np.asarray(b, np.float32)))
        out[net] = layers
    report = dict(
        new_inputs=[n for n in obs_names if n not in set(old_obs)],
        dropped_inputs=[n for n in old_obs if n not in set(obs_names)],
        new_actions=[n for n in act_names if n not in set(old_act)],
        dropped_actions=[n for n in old_act if n not in set(act_names)])
    return out, _norm(ck["onorm"], old_obs, obs_names), _norm(ck["cnorm"], old_cobs, cobs_names), report
