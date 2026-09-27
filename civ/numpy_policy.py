"""Framework-free actors: run a trained policy with nothing but numpy.

Kept apart from the JAX training code (civ/mappo.py, civ/mappo_rnn.py re-export these) so the
replay recorder and the live sandbox can run where JAX is not available, e.g. in the browser
under Pyodide.
"""
from __future__ import annotations

import json

import numpy as np


class NumpyPolicy:
    """Framework-free actor for replays / deployment."""

    def __init__(self, path):
        z = np.load(path)
        self.nl = int(z["n_layers"])
        self.W = [z[f"W{i}"] for i in range(self.nl)]
        self.b = [z[f"b{i}"] for i in range(self.nl)]
        self.mean, self.std, self.clip = z["obs_mean"], z["obs_std"], float(z["obs_clip"])
        self.meta = json.loads(bytes(z["meta_json"]).decode())

    def logits(self, obs):
        x = np.clip((obs - self.mean) / self.std, -self.clip, self.clip)
        for i in range(self.nl - 1):
            x = x @ self.W[i] + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x @ self.W[-1] + self.b[-1]

    def sample(self, obs, rng):
        lg = self.logits(obs)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        u = rng.random((len(p), 1))
        return np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)   # float rounding can leave the cdf just below u


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
