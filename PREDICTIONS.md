# Predictions — frozen before any model was trained

## Question
Sequences overlap: many share the same prefix and middle, and only a context cue says which
ending is right. They are learned one phase after another, with no replay. Does an
anatomy-shaped microcircuit (apical context gate with slow SST-like inhibition, fast PV-like
normalization at the soma, AIS-like release threshold) keep old sequences better than a plain
GRU and than a GRU with a generic context gate? Known: context gating reduces forgetting
(Masse et al. 2018; Iyer et al. 2022). The real question is whether the anatomical specifics
add anything beyond a generic gate.

## World
20 tokens. 5 phases. Each phase has 3 pairs of sequences. Every sequence is 2 prefix tokens,
a 4-token middle shared within the pair, and 3 ending tokens. Prefixes and middles are the
**same in every phase**; endings are drawn anew per phase. So the ending depends on
(prefix, phase), the middle hides the prefix for 4 steps, and later phases directly conflict
with earlier ones. The phase id (context) is given to every model on every step.

## Arms (hidden sizes chosen so parameter counts are within 15 % of each other)
- **GRU**: context as an ordinary input.
- **GATE**: GRU; a learned sigmoid gate from the context multiplies the hidden state before
  readout and recurrence (generic context gate).
- **MASSE**: GRU; a fixed random binary mask per context (50 % of units on) multiplies the
  hidden state (Masse-style, not learned).
- **MICRO**: basal GRU state h; apical gate g = σ(W_c c + W_h h − s), where s is a slow
  SST-like inhibition (time constant 4 steps) driven by the cell's own activity; PV-like fast
  divisive normalization; AIS-like graded release r = σ(k(z − χ)) with learned per-unit
  thresholds χ; the released output feeds the readout and the next step.

Training: per phase 400 Adam steps (lr 3e-3), batch 64, sequences of the current phase only.
Loss: next-token cross-entropy on every position. Seeds: 1, 2, 3, 4, 5.

## Metrics (after all 5 phases)
- **JUNCTION**: accuracy on the first ending token, averaged over all 5 phases (retention).
- **FINAL**: JUNCTION on phase 5 only (learning the newest).
- **ENDING**: accuracy on all 3 ending tokens, averaged over phases.

## Gates
- **G0** every arm reaches FINAL ≥ 0.90 (else void).
- **H1 (anatomy earns its keep)**: MICRO − GATE JUNCTION ≥ +0.05, positive in ≥ 4/5 seeds.
- **H2 (known effect)**: max(GATE, MASSE) − GRU JUNCTION ≥ +0.10.
- **H3**: MICRO − GRU JUNCTION ≥ +0.10.

Claude's guesses: H1 20 %, H2 60 %, H3 55 %. If H1 fails, the anatomical labels add nothing
beyond "a context gate" in this task.

## Ledger
(none: nothing changed after freezing)

## Addendum v0.2 — frozen before the new arm was run (written after v0's results)

v0 showed that only a fixed, unlearned context mask protected old sequences. The cheap test:
is that because of the fixed wiring, or can the anatomy help once its context wiring is fixed?

New arm **MICRO_FIXED**: identical to MICRO, except the apical context weights are a fixed random
mapping (each unit gets +4 or −4 per context, bias 0) and are never trained. Everything else in
MICRO (state-driven apical term, SST, PV, AIS) still learns. Same seeds, steps and metrics.
The four v0 arms are not re-run; their receipts stand.

- **H4 — fixed wiring rescues the microcircuit**: MICRO_FIXED − MICRO JUNCTION ≥ +0.10.
- **H5 — anatomy adds nothing beyond fixed wiring**: |MICRO_FIXED − MASSE| JUNCTION < 0.05.
  (If MICRO_FIXED beats MASSE by ≥ 0.05, the anatomy helps on top of fixed wiring; if it falls
  short by ≥ 0.05, its learned parts undo some of the protection.)

Claude's guesses: H4 65 %, H5 40 %. Risk: the learned state-driven apical term and SST can
override the fixed context drive and bring forgetting back.
