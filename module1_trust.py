"""MODULE 1 - Secure device authentication & continuous trust scoring (Eqs. 1-5)."""
import numpy as np

ALPHA, BETA, GAMMA = .3, .4, .3        # Eq. 1 weights (sum to 1)
TH_HIGH, TH_LOW = .75, .45             # Eq. 5 trust-state thresholds

class TrustEngine:
    def __init__(self, devices, dev_baseline):
        self.devices = devices                       # {id: dict(name, criticality, compliance)}
        self.dev_b = dev_baseline                    # mean |z| of benign traffic (calibrated in train.py)
        self.T = {d: 1.0 for d in devices}
    def auth_factor(self, success, attempts): return success / max(1, attempts)                 # Eq. 2
    def behaviour_factor(self, z_row):                                                         # Eq. 3
        return 1 - min(1.0, float(np.mean(np.abs(z_row))) / (6*self.dev_b))
    def compliance_factor(self, checks): return sum(checks) / len(checks)                        # Eq. 4
    def update(self, d, A, B, C):
        inst = ALPHA*A + BETA*B + GAMMA*C                                                       # Eq. 1
        self.T[d] = .5*self.T[d] + .5*inst                                                      # continuous re-scoring
        return self.T[d]
    def penalise(self, d, delta=.15):                                                           # Eq. 26 (from Module 5)
        self.T[d] = max(0.0, self.T[d] - delta); return self.T[d]
    def state(self, d):                                                                         # Eq. 5
        t = self.T[d]; return "Trusted" if t >= TH_HIGH else ("Restricted" if t >= TH_LOW else "Quarantined")
