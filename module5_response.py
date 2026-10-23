"""MODULE 5 - Response, alerting & adaptive feedback (Eqs. 25-27)."""
W = (.5, .3, .2)                                  # Eq. 25 weights
SEV = {"Benign": 0, "Reconnaissance": .5, "Spoofing": .7, "MITM_Spoofing": .8, "MQTT_Attack": .8,
       "Ransomware_C2": .9, "DoS": 1.0, "DDoS": 1.0}

class ResponseEngine:
    def __init__(self, batch=15): self.bx, self.by, self.batch, self.alerts, self.updates = [], [], batch, 0, 0
    def priority(self, conf, cls, crit): return W[0]*conf + W[1]*SEV.get(cls, .7) + W[2]*crit    # Eq. 25
    def confirmed(self, cls, conf): return cls != "Benign" and conf > .8
    def alert(self, cls, prio):
        if cls != "Benign" and prio > .6: self.alerts += 1; return f"ALERT P={prio:.2f}"
        return ""
    def collect(self, x_selected, label_idx): self.bx.append(x_selected); self.by.append(label_idx)
    def maybe_update(self, detector):                                                            # feeds Eq. 24 in Module 4
        if len(self.bx) >= self.batch:
            n = len(self.bx); detector.partial_update(self.bx, self.by); self.bx, self.by = [], []; self.updates += 1
            return n
        return 0
