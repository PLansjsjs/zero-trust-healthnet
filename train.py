"""Train + evaluate. Produces model, graphs and tables. Run:  python train.py"""
import os, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
import joblib, config, data
from module2_preprocess import Preprocessor
from module4_detector import Detector

sns.set_style("whitegrid"); os.makedirs(f"{config.OUT_DIR}/figs", exist_ok=True)
FIG = lambda n: f"{config.OUT_DIR}/figs/{n}"
Xdf, ynames, split = data.load(); feats = list(Xdf.columns)
classes = sorted(str(c) for c in set(ynames)); cidx = {c: i for i, c in enumerate(classes)}; ynames = [str(c) for c in ynames]
y = np.array([cidx[c] for c in ynames]); X = Xdf.values.astype(float)
print(f"Dataset: {X.shape[0]} flows, {X.shape[1]} features, classes: {classes}")
official = False
if split is not None:
    trm = np.array([s != "test" for s in split]); tem = ~trm
    official = all(((y == i) & trm).sum() > 0 and ((y == i) & tem).sum() > 0 for i in range(len(classes)))
if official:
    print("Using the dataset's official train/test files as the split (no leakage between windows).")
    Xtr, ytr, Xte, yte = X[trm], y[trm], X[tem], y[tem]
else:
    print("NOTE: using a random stratified split. For real data this can inflate scores (neighbouring windows from the same capture land in both sets); "
          "download the matching _train and _test files for every class to use the official split.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=config.SEED)
print("Rows per class (train/test):", {c: (int((ytr == i).sum()), int((yte == i).sum())) for i, c in enumerate(classes)})

pre = Preprocessor(config.K_FEATURES).fit(Xtr, ytr)
Ptr, Pte = pre.transform(Xtr), pre.transform(Xte)
sel_names = pre.selected(feats); print("Selected features:", sel_names)

def metrics(yt, yp):
    return dict(Accuracy=accuracy_score(yt, yp), Precision=precision_score(yt, yp, average="macro", zero_division=0),
                Recall=recall_score(yt, yp, average="macro", zero_division=0), F1=f1_score(yt, yp, average="macro", zero_division=0))

# ---- proposed model + baselines ----
t0 = time.time()
prop = Detector(len(classes), backend=config.BACKEND, epochs=config.EPOCHS, lr=3e-3, hidden=128, seed=config.SEED).fit(Ptr, ytr)
print(f"Proposed detector backend: {prop.backend}  (trained in {time.time()-t0:.1f}s)")
pname = "Proposed (CNN-BiLSTM-Attention)" if prop.backend == "torch" else "Proposed (Adaptive MLP)"
base = {"Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=config.SEED), "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(7), "SVM (RBF)": SVC(random_state=config.SEED),
        "Random Forest": RandomForestClassifier(200, max_depth=14, random_state=config.SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=config.SEED)}
rows = [dict(Model=pname, **metrics(yte, prop.predict(Pte)))]; fitted = {}
for n, m in base.items():
    m.fit(Ptr, ytr); fitted[n] = m; rows.append(dict(Model=n, **metrics(yte, m.predict(Pte))))
res = pd.DataFrame(rows).sort_values("F1", ascending=False); res.to_csv(f"{config.OUT_DIR}/model_comparison.csv", index=False)
print("\nMODEL COMPARISON\n", res.round(4).to_string(index=False))

# ---- overfitting check ----
tr_acc, te_acc = accuracy_score(ytr, prop.predict(Ptr)), accuracy_score(yte, prop.predict(Pte))
print(f"\nOverfitting check (proposed): train acc {tr_acc:.4f} vs test acc {te_acc:.4f}  (gap {tr_acc-te_acc:+.4f})")

# ---- graphs (9) ----
o = res.sort_values("Accuracy"); plt.figure(figsize=(8,5))
plt.barh(o.Model, o.Accuracy, color=["#C55A11" if m == pname else "#4472C4" for m in o.Model]); plt.xlim(0,1)
plt.title("Fig. 2 Accuracy: proposed vs baselines"); plt.tight_layout(); plt.savefig(FIG("fig2_accuracy.png")); plt.close()
x = np.arange(len(res)); w = .25; plt.figure(figsize=(9,5))
for k, (c, col) in enumerate([("Precision","#70AD47"),("Recall","#FFC000"),("F1","#4472C4")]): plt.bar(x+(k-1)*w, res[c], w, label=c, color=col)
plt.xticks(x, res.Model, rotation=30, ha="right"); plt.ylim(0,1); plt.legend(); plt.title("Fig. 3 Precision / Recall / F1")
plt.tight_layout(); plt.savefig(FIG("fig3_prf.png")); plt.close()
yp = prop.predict(Pte); cm = confusion_matrix(yte, yp, labels=range(len(classes)))
plt.figure(figsize=(7,6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Fig. 4 Confusion matrix (proposed)"); plt.tight_layout(); plt.savefig(FIG("fig4_confusion.png")); plt.close()
rep = classification_report(yte, yp, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
plt.figure(figsize=(8,4.5)); plt.bar(classes, [rep[c]["f1-score"] for c in classes], color="#4472C4"); plt.ylim(0,1.05)
plt.xticks(rotation=25, ha="right"); plt.title("Fig. 5 Per-class F1 (proposed)"); plt.tight_layout(); plt.savefig(FIG("fig5_perclass_f1.png")); plt.close()
Yb = label_binarize(yte, classes=range(len(classes))); S = prop.predict_proba(Pte); plt.figure(figsize=(7,6))
for i, c in enumerate(classes):
    if Yb[:, i].sum() == 0: continue
    f, t, _ = roc_curve(Yb[:, i], S[:, i]); plt.plot(f, t, label=f"{c} (AUC={auc(f,t):.3f})")
plt.plot([0,1],[0,1],"k--",lw=.8); plt.legend(fontsize=7); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Fig. 6 ROC (one-vs-rest)")
plt.tight_layout(); plt.savefig(FIG("fig6_roc.png")); plt.close()
plt.figure(figsize=(7,4.5)); plt.plot(prop.loss_curve, color="#C55A11"); plt.xlabel("Epoch"); plt.ylabel("Training loss")
plt.title("Fig. 7 Training loss"); plt.tight_layout(); plt.savefig(FIG("fig7_loss.png")); plt.close()
rf = fitted["Random Forest"]; oi = np.argsort(rf.feature_importances_)[-15:]; plt.figure(figsize=(7,5))
plt.barh(np.array(sel_names)[oi], rf.feature_importances_[oi], color="#70AD47"); plt.title("Fig. 8 Feature importance (Random Forest)")
plt.tight_layout(); plt.savefig(FIG("fig8_importance.png")); plt.close()
mo = np.argsort(pre.mi)[-15:]; plt.figure(figsize=(7,5)); plt.barh(np.array(feats)[mo], pre.mi[mo], color="#4472C4")
plt.title("Fig. 9 Mutual-information ranking (Eq. 8)"); plt.tight_layout(); plt.savefig(FIG("fig9_mi.png")); plt.close()

# ---- ablation ----
def variant(cols_tr, cols_te):
    d = Detector(len(classes), backend=prop.backend, epochs=config.EPOCHS, seed=config.SEED).fit(cols_tr, ytr)
    return metrics(yte, d.predict(cols_te))
Ztr, Zte = pre.scale(Xtr), pre.scale(Xte)
abl = [dict(Variant="Full proposed (MI-selected features)", **metrics(yte, prop.predict(Pte))),
       dict(Variant="No feature selection (all features)", **variant(Ztr, Zte))]
keep = [i for i, f in enumerate(feats) if not any(k in f.lower() for k in ("iat", "ttl"))]
if len(keep) < len(feats): abl.append(dict(Variant="Without timing/behavioural features (iat, ttl)", **variant(Ztr[:, keep], Zte[:, keep])))
ab = pd.DataFrame(abl); ab.to_csv(f"{config.OUT_DIR}/ablation.csv", index=False); print("\nABLATION\n", ab.round(4).to_string(index=False))

# ---- hyperparameter tuning (hold-out validation) ----
Xa, Xv, ya, yv = train_test_split(Ptr, ytr, test_size=.2, stratify=ytr, random_state=config.SEED); tune = []
for lr in (3e-3, 1e-2):
    for hid in (32, 64, 128):
        d = Detector(len(classes), backend=prop.backend, epochs=max(8, config.EPOCHS//2), lr=lr, hidden=hid, seed=config.SEED).fit(Xa, ya)
        tune.append(dict(LearningRate=lr, Hidden=hid, ValF1=f1_score(yv, d.predict(Xv), average="macro")))
tu = pd.DataFrame(tune).sort_values("ValF1", ascending=False); tu.to_csv(f"{config.OUT_DIR}/tuning.csv", index=False)
print("\nHYPERPARAMETER TUNING\n", tu.round(4).to_string(index=False))

# ---- save artefacts for main.py ----
dev_b = float(np.mean(np.abs(Ztr[ytr == cidx.get("Benign", 0)])))
prop.save(f"{config.OUT_DIR}/detector.bin"); joblib.dump(pre, f"{config.OUT_DIR}/preprocessor.joblib")
np.savez(f"{config.OUT_DIR}/test_stream.npz", X=Xte, y=yte)
json.dump(dict(classes=classes, features=feats, dev_baseline=dev_b, backend=prop.backend), open(f"{config.OUT_DIR}/meta.json", "w"))
print(f"\nSaved model, tables and graphs to ./{config.OUT_DIR}/   Next: python main.py")
