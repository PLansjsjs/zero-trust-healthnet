# Zero-Trust HealthNet - full implementation

## Run (Windows / VS Code terminal, inside this folder)
    pip install numpy pandas scikit-learn matplotlib seaborn joblib
    pip install torch                # optional but needed for the real CNN-BiLSTM-Attention model
    python train.py                  # trains, evaluates, saves model + 9 graphs + tables into outputs/
    python main.py                   # runs the whole 5-module system on a device stream
    python main.py --delay 0.2       # slower, for a live demo

Without PyTorch everything still runs, using an MLP in place of the CNN-BiLSTM-Attention model
(the first line of output tells you which backend is active). Say which one you used.

## Files = architecture
config.py            all settings (dataset path, backend, epochs)
data.py              synthetic dataset OR real CSV loader
module1_trust.py     Module 1  trust score (Eqs. 1-5)
module2_preprocess.py Module 2 scaling + mutual-information feature selection (Eqs. 7-8)
module3_policy.py    Module 3  risk score, ALLOW/RESTRICT/BLOCK, adaptive threshold (Eqs. 10-12, Alg. 1)
module4_detector.py  Module 4  CNN-BiLSTM-Attention detector + incremental update (Eqs. 13-24, Alg. 2)
module5_response.py  Module 5  alert priority, trust penalty, feedback batches (Eqs. 25-27)
train.py             Section 4 results: comparison table, 9 graphs, ablation, tuning, overfitting check
main.py              end-to-end system run + zero-trust vs detection-only comparison

## Using a real dataset (CICIoMT2024)
1. Download the CSV files, put them in one folder.
2. In config.py set DATA_SOURCE = r"C:\path\to\that\folder" and K_FEATURES = 25.
3. python train.py ; python main.py
The loader takes the label from LABEL_COLUMN if you set it, otherwise from the file name
(benign / DDoS / DoS / recon / ARP-spoofing / MQTT). If your files are named differently,
edit family() in data.py. Real data was NOT available when this was written, so check the
class list printed by train.py.

## What is simulated (say this to your teacher)
Device identities, the authentication log (factor A) and the firmware-compliance checks (factor C)
are simulated; traffic features and detections come from the dataset and the model.

## Results (full CICIoMT2024, official train/test split, 6 classes)
Gradient Boosting F1 0.955 | Random Forest 0.951 | Decision Tree 0.942 | KNN 0.822 |
Proposed CNN-BiLSTM-Attention 0.791 | SVM 0.773 | Naive Bayes 0.532

## Dataset
The full dataset is several GB, so this repo includes only data_sample/ (first 2000 rows of each CSV).
Full data: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
To use it, set DATA_SOURCE in config.py to the folder containing the CSVs.
The results above were produced on the full data. The saved outputs/ folder lets you run `python main.py` immediately.
