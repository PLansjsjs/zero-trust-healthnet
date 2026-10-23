"""Dataset loading: built-in synthetic IoMT traffic OR real CSV files (e.g. CICIoMT2024)."""
import os, glob, hashlib
import numpy as np, pandas as pd
import config

FEATS = ["flow_duration","packet_count","byte_count","flow_rate","pkt_size_mean","pkt_size_std",
         "iat_mean","ttl","syn_ratio","ack_ratio","dst_port_entropy"]
_P = {
 "Benign":[(2.5,.8),(40,10),(6000,1500),(16,4),(150,30),(20,6),(.06,.02),(64,2),(.045,.02),(.5,.06),(1.0,.3)],
 "DoS":[(.4,.2),(400,90),(24000,5000),(140,25),(60,15),(8,3),(.002,.001),(58,4),(.72,.13),(.1,.06),(.3,.15)],
 "MITM_Spoofing":[(3.2,1),(55,15),(7500,2000),(18,5),(140,35),(45,12),(.07,.03),(48,6),(.125,.04),(.45,.06),(1.1,.4)],
 "Reconnaissance":[(.15,.08),(3,1.2),(220,80),(22,8),(70,20),(10,4),(.03,.015),(64,2),(.75,.09),(.05,.03),(3.2,.6)],
 "Ransomware_C2":[(5,1.5),(25,8),(3200,900),(6,2),(110,25),(18,5),(.2,.08),(62,3),(.075,.026),(.55,.06),(.6,.25)]}

def synthetic(n_per_class, rng):
    X, y = [], []
    for c, p in _P.items():
        X.append(np.clip(np.column_stack([rng.normal(m, s, n_per_class) for m, s in p]), 0, None))
        y += [c]*n_per_class
    return pd.DataFrame(np.vstack(X), columns=FEATS), np.array(y)

def family(text):
    t = text.lower()
    if "benign" in t: return "Benign"
    if "mqtt" in t: return "MQTT_Attack"
    if "ddos" in t: return "DDoS"
    if "dos" in t: return "DoS"
    if "recon" in t or "scan" in t or "ping" in t or "os_" in t: return "Reconnaissance"
    if "arp" in t or "spoof" in t: return "Spoofing"
    return os.path.splitext(os.path.basename(text))[0]

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def real_csv(folder, max_per_class, rng):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    if not files: raise SystemExit(f"No CSV files found under {folder}")
    seen, parts = set(), []
    for f in files:
        h = _md5(f)
        if h in seen: print("  skipping duplicate file:", os.path.basename(f)); continue
        seen.add(h); name = os.path.basename(f)
        sp = "test" if "test" in name.lower() else ("train" if "train" in name.lower() else "unknown")
        cap = max_per_class if sp != "test" else max(1, max_per_class // 3)
        chunks = []
        for ch in pd.read_csv(f, chunksize=200_000, low_memory=False):      # chunked: safe for multi-million-row files
            if config.LABEL_COLUMN and config.LABEL_COLUMN in ch.columns:
                lab = ch[config.LABEL_COLUMN].astype(str).map(family).values; ch = ch.drop(columns=[config.LABEL_COLUMN])
            else:
                lab = np.array([family(name)] * len(ch))
            ch = ch.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            ch["__y"] = lab; ch["__s"] = sp
            chunks.append(ch.sample(min(len(ch), cap), random_state=config.SEED))
        d = pd.concat(chunks, ignore_index=True)
        parts.append(d.sample(min(len(d), cap), random_state=config.SEED))
    d = pd.concat(parts, ignore_index=True).drop_duplicates()
    d = d.dropna(axis=1, thresh=int(.9 * len(d))).fillna(0)
    keep = []
    for (c, s), g in d.groupby(["__y", "__s"]):
        cap = max_per_class if s != "test" else max(1, max_per_class // 3)
        keep.append(g.sample(min(len(g), cap), random_state=config.SEED))
    d = pd.concat(keep, ignore_index=True)
    y, sp = d.pop("__y").values, d.pop("__s").values
    return d, y, sp

def load():
    rng = np.random.default_rng(config.SEED)
    if config.DATA_SOURCE == "synthetic":
        X, y = synthetic(config.MAX_PER_CLASS, rng); return X, y, None
    return real_csv(config.DATA_SOURCE, config.MAX_PER_CLASS, rng)
