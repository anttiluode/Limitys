"""Overlapping sequences learned phase after phase: anatomy-shaped microcircuit vs gated GRUs.
Everything follows PREDICTIONS.md."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

V, P, PAIRS, WORLD_SEED = 20, 5, 3, 0
PRE, MID, END = 2, 4, 3
L = PRE + MID + END
ARMS = ("GRU", "GATE", "MASSE", "MICRO", "MICRO_FIXED", "MICRO_HARD")
SEEDS = (1, 2, 3, 4, 5)
CFG = {"steps_per_phase": 400, "lr": 3e-3, "batch": 64}
HIDDEN = {"GRU": 64, "GATE": 64, "MASSE": 64, "MICRO": 52, "MICRO_FIXED": 52, "MICRO_HARD": 52}


def make_world():
    """seqs[p] = (6, L) tokens. Prefixes and middles shared across phases; endings per phase."""
    rng = np.random.default_rng(WORLD_SEED)
    heads = []
    for _ in range(PAIRS):
        mid = rng.choice(V, MID, replace=False)
        for _ in range(2):
            heads.append(np.concatenate([rng.choice(V, PRE, replace=False), mid]))
    heads = np.array(heads)
    # the two prefixes of a pair must differ in their first token so the history really differs
    for j in range(PAIRS):
        while heads[2 * j, 0] == heads[2 * j + 1, 0]:
            heads[2 * j + 1, 0] = rng.integers(V)
    seqs = []
    for _ in range(P):
        ends = np.array([rng.choice(V, END, replace=False) for _ in range(2 * PAIRS)])
        # within a pair, the first ending token must differ (a real junction)
        for j in range(PAIRS):
            while ends[2 * j, 0] == ends[2 * j + 1, 0]:
                ends[2 * j + 1, 0] = rng.integers(V)
        seqs.append(np.concatenate([heads, ends], axis=1))
    return np.array(seqs)  # (P, 6, L)


class Net(nn.Module):
    def __init__(self, arm: str, seed: int):
        super().__init__()
        self.arm, H = arm, HIDDEN[arm]
        self.H = H
        micro = arm.startswith("MICRO")
        d_in = V + (0 if micro else P)
        self.cell = nn.GRUCell(d_in, H)
        self.out = nn.Linear(H, V)
        if arm == "GATE":
            self.gate = nn.Linear(P, H)
        if arm == "MASSE":
            g = torch.Generator().manual_seed(1000 + seed)
            self.register_buffer("mask", (torch.rand(P, H, generator=g) < 0.5).float())
        if micro:
            self.apical_c = nn.Linear(P, H)            # context arrives at the tuft
            self.apical_h = nn.Linear(H, H, bias=False)
            self.sst = nn.Linear(H, H)                 # slow inhibition driven by own activity
            self.chi = nn.Parameter(torch.zeros(H))    # AIS release threshold per unit
            self.k = nn.Parameter(torch.tensor(1.6))   # release steepness (softplus -> ~1.8)
        if arm == "MICRO_HARD":
            g2 = torch.Generator().manual_seed(3000 + seed)
            self.register_buffer("hard", (torch.rand(P, H, generator=g2) < 0.5).float())
        if arm in ("MICRO_FIXED", "MICRO_HARD"):
            # fixed random apical context wiring: +4 or -4 per unit and context, never trained
            g = torch.Generator().manual_seed(2000 + seed)
            w = (torch.rand(H, P, generator=g) < 0.5).float() * 8 - 4
            with torch.no_grad():
                self.apical_c.weight.copy_(w)
                self.apical_c.bias.zero_()
            self.apical_c.weight.requires_grad_(False)
            self.apical_c.bias.requires_grad_(False)

    def forward(self, tok: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, T = tok.shape
        c = F.one_hot(ctx, P).float()
        p = torch.zeros(B, self.H)
        s = torch.zeros(B, self.H)
        logits = []
        for t in range(T):
            x = F.one_hot(tok[:, t], V).float()
            if self.arm.startswith("MICRO"):
                h = self.cell(x, p)
                s = s + (F.relu(self.sst(h)) - s) / 4.0                       # SST: slow
                g = torch.sigmoid(self.apical_c(c) + self.apical_h(h) - s)   # apical gate
                if self.arm == "MICRO_HARD":
                    g = g * self.hard[ctx]                                    # hard 0/1 exclusion
                z = g * h
                z = z / (1.0 + z.abs().mean(-1, keepdim=True))               # PV: fast divisive
                r = torch.sigmoid(F.softplus(self.k) * 4 * (z - self.chi))  # AIS release
                p = r * z
            else:
                h = self.cell(torch.cat([x, c], -1), p)
                if self.arm == "GATE":
                    p = torch.sigmoid(self.gate(c)) * h
                elif self.arm == "MASSE":
                    p = self.mask[ctx] * h
                else:
                    p = h
            logits.append(self.out(p))
        return torch.stack(logits, 1)  # (B, T, V): prediction of token t+1 at position t


def evaluate(net: Net, seqs: np.ndarray) -> dict:
    per_phase_j, per_phase_e = [], []
    with torch.no_grad():
        for ph in range(P):
            tok = torch.as_tensor(seqs[ph])
            pred = net(tok[:, :-1], torch.full((len(tok),), ph)).argmax(-1)
            tgt = tok[:, 1:]
            j = PRE + MID - 1  # position predicting the first ending token
            per_phase_j.append(float((pred[:, j] == tgt[:, j]).float().mean()))
            per_phase_e.append(float((pred[:, j:] == tgt[:, j:]).float().mean()))
    return {"JUNCTION": float(np.mean(per_phase_j)), "FINAL": per_phase_j[-1],
            "ENDING": float(np.mean(per_phase_e)), "junction_by_phase": per_phase_j}


def run(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    seqs = make_world()
    net = Net(arm, seed)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    rng = np.random.default_rng(seed)
    after_phase = []
    for ph in range(P):
        for _ in range(CFG["steps_per_phase"]):
            idx = rng.integers(0, 2 * PAIRS, CFG["batch"])
            tok = torch.as_tensor(seqs[ph][idx])
            logits = net(tok[:, :-1], torch.full((len(idx),), ph))
            loss = F.cross_entropy(logits.reshape(-1, V), tok[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        after_phase.append(evaluate(net, seqs)["junction_by_phase"])
    res = evaluate(net, seqs)
    res.update({"arm": arm, "seed": seed, "params": sum(p.numel() for p in net.parameters() if p.requires_grad),
                "junction_matrix": after_phase})
    return res
