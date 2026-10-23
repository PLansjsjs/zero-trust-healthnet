"""MODULE 3 - Zero-trust policy enforcement (Eqs. 10-12, Algorithm 1)."""
import numpy as np
from module1_trust import TH_LOW

LAMBDA, ETA, R_TARGET = .5, .05, .3

class PolicyEngine:
    def __init__(self): self.theta_high = .75
    def risk(self, T, anomaly): return (1 - T) + LAMBDA*anomaly                                # Eq. 11
    def decide(self, T, anomaly):                                                              # Eq. 10 / Algorithm 1
        R = self.risk(T, anomaly)
        if T < TH_LOW or R >= .9: dec = "BLOCK"
        elif T < self.theta_high or R >= .5: dec = "RESTRICT"
        else: dec = "ALLOW"
        self.theta_high = float(np.clip(self.theta_high + ETA*(R_TARGET - R)*.1, .65, .85))    # Eq. 12 adaptive threshold
        return dec, R
