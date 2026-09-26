"""Run every (arm, seed), write receipts, then apply the frozen gates. Two processes at a time."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from limitys import ARMS, SEEDS, run

OUT = Path("results")

def job(a_s):
    a, s = a_s
    f = OUT / f"{a}_seed{s}.json"
    if not f.exists():
        f.write_text(json.dumps(run(a, s), indent=2))
    return f.name

def summarize():
    R = {(a, s): json.loads((OUT / f"{a}_seed{s}.json").read_text()) for a in ARMS for s in SEEDS}
    m = {a: {k: float(np.mean([R[(a, s)][k] for s in SEEDS])) for k in ("JUNCTION", "FINAL", "ENDING", "params")} for a in ARMS}
    d = lambda x, y: np.array([R[(x, s)]["JUNCTION"] - R[(y, s)]["JUNCTION"] for s in SEEDS])
    h1, h3 = d("MICRO", "GATE"), d("MICRO", "GRU")
    best_gate = max(m["GATE"]["JUNCTION"], m["MASSE"]["JUNCTION"])
    gates = {"G0": all(m[a]["FINAL"] >= 0.90 for a in ARMS),
             "H1": bool(h1.mean() >= 0.05 and (h1 > 0).sum() >= 4),
             "H2": bool(best_gate - m["GRU"]["JUNCTION"] >= 0.10),
             "H3": bool(h3.mean() >= 0.10)}
    out = {"gates": gates, "means": m,
           "H1_micro_minus_gate": {"mean": float(h1.mean()), "positive": int((h1 > 0).sum()), "per_seed": h1.tolist()},
           "H3_micro_minus_gru": {"mean": float(h3.mean()), "positive": int((h3 > 0).sum())},
           "H2_best_gate_minus_gru": float(best_gate - m["GRU"]["JUNCTION"])}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    if sys.argv[1:] == ["summarize"]:
        summarize()
    else:
        with ProcessPoolExecutor(2) as ex:
            for n in ex.map(job, [(a, s) for s in SEEDS for a in ARMS]):
                print("done", n, flush=True)
        summarize()
