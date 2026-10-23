"""MODULE 4 - Adaptive AI-based intrusion detector.
Backend 'torch': CNN -> BiLSTM -> Attention (Eqs. 13-22), cross-entropy (Eq. 23), incremental update (Eq. 24).
Backend 'mlp'  : scikit-learn MLP fallback (used automatically if PyTorch is not installed)."""
import numpy as np, joblib
try:
    import torch, torch.nn as nn
    HAS_TORCH = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # uses the GPU automatically if available
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    class Net(nn.Module):
        def __init__(s, n_classes, ch=32, hid=32):
            super().__init__()
            s.conv = nn.Sequential(nn.Conv1d(1, ch, 3, padding=1), nn.ReLU(),        # Eq. 13  y_i = sum_k w_k x_{i+k} + b
                                   nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
            s.lstm = nn.LSTM(ch, hid, batch_first=True, bidirectional=True)         # Eqs. 14-18 (gates, cell, hidden)
            s.att_w = nn.Linear(2*hid, 2*hid); s.att_v = nn.Linear(2*hid, 1, bias=False)
            s.out = nn.Linear(2*hid, n_classes)
        def forward(s, x):                                  # x: (batch, n_features)
            h = s.conv(x.unsqueeze(1)).transpose(1, 2)      # feature axis treated as a sequence
            h, _ = s.lstm(h)
            e = s.att_v(torch.tanh(s.att_w(h))).squeeze(-1) # Eq. 19  e_t = v^T tanh(W h_t + b)
            a = torch.softmax(e, dim=1)                     # Eq. 20  attention weights
            c = (a.unsqueeze(-1) * h).sum(1)                # Eq. 21  context vector
            return s.out(c)                                 # Eq. 22  logits -> softmax

class Detector:
    def __init__(self, n_classes, backend="auto", epochs=20, lr=1e-3, hidden=32, seed=7):
        self.n_classes, self.epochs, self.lr, self.hidden, self.seed = n_classes, epochs, lr, hidden, seed
        self.backend = ("torch" if HAS_TORCH else "mlp") if backend == "auto" else backend
        if self.backend == "torch" and not HAS_TORCH:
            raise SystemExit("BACKEND='torch' but PyTorch is not installed. Run: pip install torch")
        self.loss_curve = []

    # ---------------- training ----------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        if self.backend == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.m = MLPClassifier(hidden_layer_sizes=(self.hidden*4, self.hidden*2), learning_rate_init=self.lr*10 if self.lr<=1e-3 else self.lr,
                                   max_iter=300, random_state=self.seed).fit(X, y)
            self.loss_curve = list(self.m.loss_curve_); return self
        torch.manual_seed(self.seed)
        self.net = Net(self.n_classes, hid=self.hidden).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr); lossf = nn.CrossEntropyLoss()
        Xt, yt = torch.tensor(X).to(DEV), torch.tensor(y, dtype=torch.long).to(DEV)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(Xt), device=DEV); tot = 0
            for i in range(0, len(Xt), 128):
                idx = perm[i:i+128]; opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])                    # Eq. 23 cross-entropy
                loss.backward(); opt.step(); tot += loss.item()*len(idx)
            self.loss_curve.append(tot/len(Xt))
        return self

    # ---------------- inference ----------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.backend == "mlp":
            p = np.zeros((len(X), self.n_classes)); p[:, self.m.classes_] = self.m.predict_proba(X); return p
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):                                # chunked so large sets fit in GPU memory
                out.append(torch.softmax(self.net(torch.tensor(X[i:i+4096]).to(DEV)), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.n_classes))
    def predict(self, X): return self.predict_proba(X).argmax(1)

    # ---------------- adaptive (incremental) update ----------------
    def partial_update(self, X, y, steps=5, lr=1e-4):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        if self.backend == "mlp": self.m.partial_fit(X, y); return          # Eq. 24 (SGD step on new batch)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr); lossf = nn.CrossEntropyLoss(); self.net.train()
        Xt, yt = torch.tensor(X).to(DEV), torch.tensor(y, dtype=torch.long).to(DEV)
        for _ in range(steps):
            opt.zero_grad(); lossf(self.net(Xt), yt).backward(); opt.step()  # theta <- theta - eta * grad L

    # ---------------- persistence ----------------
    def save(self, path):
        if self.backend == "mlp": joblib.dump({"b": "mlp", "m": self.m, "n": self.n_classes}, path)
        else: torch.save({"b": "torch", "s": self.net.state_dict(), "n": self.n_classes, "h": self.hidden}, path)
    @staticmethod
    def load(path):
        try: d = joblib.load(path)
        except Exception: d = None
        if d is not None and d.get("b") == "mlp":
            det = Detector(d["n"], backend="mlp"); det.m = d["m"]; return det
        d = torch.load(path, weights_only=False, map_location="cpu")
        det = Detector(d["n"], backend="torch", hidden=d["h"]); det.net = Net(d["n"], hid=d["h"]); det.net.load_state_dict(d["s"]); det.net.to(DEV); return det
