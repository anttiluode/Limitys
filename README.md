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

## v0.2: the microcircuit with fixed context wiring

Pre-registered in the `PREDICTIONS.md` addendum before it ran. MICRO_FIXED is MICRO with its apical
context weights replaced by a fixed random ±4 per unit and context, never trained. Everything
else still learns. Same 5 seeds.

| arm | retention, all phases | all ending tokens |
|---|---:|---:|
| MICRO (learned context wiring) | 0.233 | 0.309 |
| MICRO_FIXED | 0.273 | 0.387 |
| MASSE (fixed 0/1 mask) | 0.793 | 0.782 |

| gate | result |
|---|---|
| H4 fixed wiring rescues the microcircuit (≥ +0.10 over MICRO) | **fail**: +0.040 (2/5 seeds positive) |
| H5 MICRO_FIXED ties MASSE (within 0.05) | **fail**: 0.52 short, all 5 seeds |

Fixing the context wiring was not enough. The microcircuit's own learned parts (the
state-driven apical term, SST inhibition) sit in the same gate as the fixed context drive, so
each new phase can learn to override it, and forgetting came back almost fully.

What separates MASSE from MICRO_FIXED, as an untested reading: MASSE's mask is hard. A unit that
is off for a context outputs exactly zero, so training on that context sends it no gradient and
cannot touch it. MICRO_FIXED's gate is a sigmoid that is never exactly zero and can be pushed by
learned terms, so gradients from new phases still reach the old units. If this is right, the
protection comes from **hard, unlearnable exclusion**, not from where the gate sits or from
fixed wiring as such. The direct test is v0.3, below.

## v0.3: the microcircuit with a hard 0/1 apical gate

Pre-registered before it ran. MICRO_HARD is MICRO_FIXED with a fixed hard 0/1 mask per context
multiplied onto its apical gate: units that are off for a context output exactly zero.

| arm | retention, all phases | all ending tokens |
|---|---:|---:|
| MICRO_FIXED (soft fixed gate) | 0.273 | 0.387 |
| MICRO_HARD (hard 0/1 gate) | **0.533** | 0.669 |
| MASSE (hard 0/1 mask on a plain GRU) | 0.793 | 0.782 |

| gate | result |
|---|---|
| H6 hard exclusion rescues the microcircuit (≥ +0.20 over MICRO_FIXED) | **pass**: +0.260, 5/5 seeds |
| H7 MICRO_HARD ties MASSE (within 0.05) | **fail**: 0.260 short, all 5 seeds |

So hard exclusion is what protects old sequences: it roughly doubled the microcircuit's
retention. But the anatomical microcircuit with the same kind of mask is still clearly *worse*
than a plain GRU with it. An untested reading: the microcircuit's extra learned parts (SST
drive, PV normalization, AIS release thresholds) are shared by every context, so each new phase
still retunes them and disturbs the old ones.

## Bottom line across v0–v0.3

1. Learned gates, anatomical or generic, did not protect old sequences at all.
2. What protected them was hard exclusion: units that are exactly off for a context, so new
   learning cannot reach them.
3. The anatomical decomposition added nothing on top, and with hard exclusion it did worse
   than a plain gated GRU.

## Limits

- One task family, 5 seeds, small networks, 400 steps per phase, no weight stabilization.
- MICRO is one hand-made reading of the anatomy; others might do better.
- v0.2 and v0.3 were designed after seeing earlier results, each pre-registered before it ran.
- Not new: context-dependent gating against forgetting (Masse, Grant & Freedman 2018);
  active dendrites (Iyer et al. 2022). What is specific here is the direct comparison
  showing the anatomical specifics did not help.

## Run it

```
pip install numpy torch
python run.py            # 30 runs + summary, ~4 min on 2 cores
```
