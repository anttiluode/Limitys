# Limitys

*Overlap.* Many sequences share the same beginning and middle; only a context cue says which
ending is right. They are learned one phase after another, with no replay. Does an
anatomy-shaped microcircuit keep the old ones better than a GRU with a generic context gate?

**Answer: no.** The anatomical microcircuit forgot exactly as much as a plain GRU. The only
thing that protected old sequences was a *fixed, unlearned* random gate per context.

Predictions, arms and thresholds were committed before any model existed (`PREDICTIONS.md`).

## Setup

- 20 tokens, 5 phases, 6 sequences per phase: 2 prefix tokens, a 4-token middle shared within
  a pair, 3 ending tokens. Prefixes and middles are identical in every phase; endings change.
  So each ending depends on (prefix, phase), the middle hides the prefix for 4 steps, and every
  new phase directly contradicts the old ones.
- Every model sees the phase id on every step. ~18.4–19.2k parameters each (within 4 %).
- 400 Adam steps per phase, current phase only. 5 seeds. Two minutes on two cores.

| arm | how context enters |
|---|---|
| GRU | as an ordinary input |
| GATE | learned sigmoid gate on the hidden state |
| MASSE | fixed random 50 % unit mask per context (Masse et al. 2018), never learned |
| MICRO | at the apical tuft: gate σ(context + state − slow SST-like inhibition), then PV-like divisive normalization, then an AIS-like graded release threshold |

## Result (`results/summary.json`)

Accuracy on the first ending token, the junction, after all 5 phases:

| arm | retention, all phases | newest phase | all ending tokens |
|---|---:|---:|---:|
| GRU | 0.233 | 1.00 | 0.271 |
| GATE | 0.233 | 1.00 | 0.296 |
| MASSE | **0.793** | 1.00 | **0.782** |
| MICRO | 0.233 | 1.00 | 0.309 |

| gate | result |
|---|---|
| G0 every arm learns the newest phase (≥ 0.90) | pass, all 1.00 |
| H1 anatomy beats a generic gate (MICRO − GATE ≥ +0.05, 4/5 seeds) | **fail**: 0.000, 0/5 |
| H2 a context gate beats a plain GRU (≥ +0.10) | pass: +0.56 (the fixed gate) |
| H3 MICRO beats a plain GRU (≥ +0.10) | **fail**: 0.000 |

0.233 is what you get from knowing only the newest phase: every learned arm overwrote the old
endings completely.

## What it means

- **The anatomical labels added nothing here.** An apical context gate with slow SST-like
  inhibition, PV-like normalization and an AIS-like release behaved exactly like a learned
  gate and exactly like a plain GRU.
- **What protects old sequences is not gating as such but gating that learning cannot move.**
  A learned gate, anatomical or not, is itself retrained by each new phase and stops
  separating anything. The fixed random mask gives each context its own units that later
  phases cannot take over.
- **For the "where does sequencing live" question:** in this task, the part that mattered
  was *which units a context is allowed to use*, and that assignment had to be stable. That
  points away from fast inhibitory control and toward slow, protected routing (in
  biology: wiring, or plasticity that is itself gated), which is Masse et al.'s conclusion too.

## Limits

- One task family, 5 seeds, small networks, 400 steps per phase, no weight stabilization.
- MICRO is one hand-made reading of the anatomy. A version whose apical context mapping is
  fixed rather than learned would probably behave like MASSE; that was not run, and if it
  does, the credit goes to the fixed mapping, not to the anatomy.
- Not new: context-dependent gating against forgetting (Masse, Grant & Freedman 2018);
  active dendrites (Iyer et al. 2022). What is specific here is the direct comparison
  showing the anatomical specifics did not help.

## Run it

```
pip install numpy torch
python run.py            # 20 runs + summary, ~2 min on 2 cores
```
