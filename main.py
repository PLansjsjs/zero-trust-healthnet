"""Zero-Trust HealthNet - full system run (Modules 1-5). First run:  python train.py   then:  python main.py
Options:  python main.py --delay 0.2     (slow the stream down for a live demo)"""
import json, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import joblib, config
from module1_trust import TrustEngine
from module3_policy import PolicyEngine
from module4_detector import Detector
from module5_response import ResponseEngine, SEV

ap = argparse.ArgumentParser(); ap.add_argument("--delay", type=float, default=0.0); ap.add_argument("--steps", type=int, default=50)
args = ap.parse_args(); rng = np.random.default_rng(config.SEED)
try:
    meta = json.load(open(f"{config.OUT_DIR}/meta.json"))
except FileNotFoundError:
    raise SystemExit("Model not found - run  python train.py  first.")
classes = meta["classes"]; pre = joblib.load(f"{config.OUT_DIR}/preprocessor.joblib"); det = Detector.load(f"{config.OUT_DIR}/detector.bin")
S = np.load(f"{config.OUT_DIR}/test_stream.npz"); pool = {c: S["X"][S["y"] == i] for i, c in enumerate(classes)}
BEN = "Benign" if "Benign" in classes else classes[0]
attacks = sorted([c for c in classes if c != BEN and len(pool[c])], key=lambda c: SEV.get(c, .7))
DEVICES = {"D1": dict(name="Patient monitor", crit=.9, comp=[1,1,1,1]), "D2": dict(name="Infusion pump", crit=1.0, comp=[1,0,1,0]),
           "D3": dict(name="MRI console", crit=.8, comp=[1,1,1,1]), "D4": dict(name="Wearable gateway", crit=.6, comp=[1,1,1,1]),
           "D5": dict(name="Ventilator", crit=1.0, comp=[1,1,1,1]), "D6": dict(name="Lab analyzer", crit=.5, comp=[1,1,1,1])}
BAD, T_ATT = "D4", 15
script = [attacks[0]]*6 + [attacks[min(1, len(attacks)-1)]]*10 + [attacks[-1]]*(args.steps)

trust = TrustEngine(DEVICES, meta["dev_baseline"]); policy = PolicyEngine(); resp = ResponseEngine()
print("="*78 + f"\nZERO-TRUST HEALTHNET  |  detector backend: {det.backend}  |  classes: {classes}\n" + "="*78)
print("Time | Device          | Predicted (conf)         | Anom | Trust | Risk | Decision | Alert\n" + "-"*100)
hist = {d: [] for d in DEVICES}; prev = {d: "ALLOW" for d in DEVICES}; lat = []
zt = dict(atk_total=0, atk_pass=0, ben_total=0, ben_block=0, ben_restrict=0)
do = dict(atk_pass=0, ben_block=0); fp = tn = 0; correct = total = 0

for t in range(args.steps):
    for d, info in DEVICES.items():
        attacking = d == BAD and t >= T_ATT; true = script[t-T_ATT] if attacking else BEN
        x = pool[true][rng.integers(len(pool[true]))][None, :]
        t0 = time.perf_counter()
        z = pre.scale(x); xs = pre.transform(x)                                       # Module 2
        p = det.predict_proba(xs)[0]; k = int(p.argmax()); pred = classes[k]; conf = float(p[k])    # Module 4
        anom = 1 - float(p[classes.index(BEN)])
        A = trust.auth_factor(97, 100) if not attacking else trust.auth_factor(max(30, 97-6*(t-T_ATT+1)), 100)  # simulated auth log
        T = trust.update(d, A, trust.behaviour_factor(z[0]), trust.compliance_factor(info["comp"]))             # Module 1
        confirmed = resp.confirmed(pred, conf)
        if confirmed: T = trust.penalise(d)                                                                   # Module 5 -> Module 1
        dec, R = policy.decide(T, anom)                                                                      # Module 3
        lat.append((time.perf_counter()-t0)*1000)
        alert = resp.alert(pred, resp.priority(conf, pred, info["crit"]))                                    # Module 5
        if confirmed: resp.collect(xs[0], k)
        # --- bookkeeping: zero-trust vs detection-only ---
        if true == BEN:
            zt["ben_total"] += 1; zt["ben_block"] += dec == "BLOCK"; zt["ben_restrict"] += dec == "RESTRICT"
            do["ben_block"] += pred != BEN; fp += pred != BEN; tn += pred == BEN
        else:
            zt["atk_total"] += 1; zt["atk_pass"] += dec == "ALLOW"; do["atk_pass"] += pred == BEN
        correct += pred == true; total += 1; hist[d].append(T)
        show = (d == BAD and (t % 3 == 0 or alert or dec != prev[d])) or (d != BAD and (alert or dec != prev[d] or t % 10 == 0))
        if show: print(f"t={t:<3}| {d} {info['name'][:11]:<11} | {pred:<16}({conf:.2f})      | {anom:.2f} | {T:.2f}  | {R:.2f} | {dec:<8} | {alert}")
        prev[d] = dec
        if args.delay: time.sleep(args.delay)
    n = resp.maybe_update(det)
    if n: print(f"t={t:<3}| [Module 4/5] incremental model update on {n} confirmed samples (update #{resp.updates})")

print("\n" + "="*78 + "\nRESULTS\n" + "="*78)
print("Final trust: " + ", ".join(f"{d}={trust.T[d]:.2f}" for d in DEVICES))
print(f"Compromised device {BAD}: final decision = {prev[BAD]}   |   healthy devices blocked: {[d for d in DEVICES if d!=BAD and prev[d]=='BLOCK'] or 'none'}")
print(f"Alerts: {resp.alerts} | model updates: {resp.updates} | mean per-flow latency: {np.mean(lat):.2f} ms | stream accuracy: {correct/total:.4f}")
print(f"False-positive rate on benign flows (Eq. 27): {fp/max(1,fp+tn):.4f}")
print("\nSYSTEM-LEVEL COMPARISON (what zero-trust adds over a stand-alone detector)")
print(f"{'':34}{'Detection-only':>16}{'Zero-Trust HealthNet':>24}")
print(f"{'Attack flows let through':34}{do['atk_pass']:>16}{zt['atk_pass']:>24}   (of {zt['atk_total']})")
print(f"{'Benign flows blocked':34}{do['ben_block']:>16}{zt['ben_block']:>24}   (of {zt['ben_total']})")
print(f"{'Benign flows restricted (extra)':34}{'-':>16}{zt['ben_restrict']:>24}")
plt.figure(figsize=(9,5))
for d, info in DEVICES.items(): plt.plot(hist[d], label=f"{d} {info['name']}", lw=3 if d == BAD else 1.2)
plt.axhline(.75, ls="--", c="orange", lw=.8); plt.axhline(.45, ls="--", c="red", lw=.8); plt.axvline(T_ATT, ls=":", c="k")
plt.xlabel("Time step"); plt.ylabel("Trust score T_d(t)"); plt.ylim(0, 1.05); plt.legend(fontsize=7); plt.title("Trust score over time (restrict < 0.75, block < 0.45)")
plt.tight_layout(); plt.savefig(f"{config.OUT_DIR}/figs/trust_over_time.png", dpi=150); print(f"\nSaved {config.OUT_DIR}/figs/trust_over_time.png")
