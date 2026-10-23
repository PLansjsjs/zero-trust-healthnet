# Zero-Trust HealthNet: Adaptive AI-Based Intrusion Detection for Connected Healthcare Infrastructure

Five-module zero-trust system (trust scoring, preprocessing, policy engine, CNN-BiLSTM-Attention detector, response) evaluated on CICIoMT2024.

## Run (Windows / VS Code terminal, inside this folder)

```
pip install numpy pandas scikit-learn matplotlib seaborn joblib torch
python main.py --delay 0.2    # live demo using the saved model in outputs/
python train.py               # retrains, evaluates, saves model + graphs + tables into outputs/
```

The first line of output shows the active backend (`torch` = CNN-BiLSTM-Attention, `mlp` = fallback if PyTorch is missing).
Note: `python train.py` overwrites `outputs/`. With `DATA_SOURCE = "data_sample"` it trains on the small sample only.

## Files = architecture

```
config.py              settings (dataset path, backend, epochs)
data.py                CSV loader (label from file name; official train/test split)
module1_trust.py       Module 1  trust score (Eqs. 1-5)
module2_preprocess.py  Module 2  scaling + mutual-information feature selection (Eqs. 7-8)
module3_policy.py      Module 3  risk score, ALLOW/RESTRICT/BLOCK, adaptive threshold (Eqs. 10-12, Alg. 1)
module4_detector.py    Module 4  CNN-BiLSTM-Attention detector + incremental update (Eqs. 13-24, Alg. 2)
module5_response.py    Module 5  alert priority, trust penalty, feedback batches (Eqs. 25-27)
train.py               results: comparison table, 9 graphs, ablation, tuning, overfitting check
main.py                end-to-end system run + zero-trust vs detection-only comparison
```

## Results (full CICIoMT2024, official train/test split, 6 classes)

| Model | F1 |
|---|---|
| Gradient Boosting | 0.955 |
| Random Forest | 0.951 |
| Decision Tree | 0.942 |
| KNN | 0.822 |
| Proposed CNN-BiLSTM-Attention | 0.791 |
| SVM (RBF) | 0.773 |
| Naive Bayes | 0.532 |

Classes: Benign, DDoS, DoS, MQTT_Attack, Reconnaissance, Spoofing. Graphs and tables are in `outputs/`.

## Dataset

The full dataset is several GB, so this repo includes only `data_sample/` (first 2000 rows of each CSV).
Full data: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
To use it, put the CSV files in one folder and set `DATA_SOURCE = r"C:\path\to\folder"` in `config.py`.
The loader takes the label from the file name. The results above were produced on the full data.

## What is simulated

Device identities, the authentication log (factor A) and the firmware-compliance checks (factor C) are simulated.
Traffic features and detections come from the CICIoMT2024 dataset and the trained model.