"""Central settings for Zero-Trust HealthNet. Edit this file only."""
DATA_SOURCE = r"data_sample"   #"C:\path\to\CICIoMT2024"  (folder containing the CSV files)
LABEL_COLUMN = None            # name of the label column in your CSVs; None = infer label from file name
BACKEND = "auto"               # "auto" = CNN-BiLSTM-Attention if PyTorch is installed, else MLP fallback
                               # "torch" / "mlp" to force one
K_FEATURES = 25                # features kept by mutual-information selection (use ~20 for real data)
MAX_PER_CLASS = 10000           # cap rows per class (keeps training fast on a laptop)
EPOCHS = 60                    # training epochs for the CNN-BiLSTM-Attention model
SEED = 7
OUT_DIR = "outputs"            # graphs, tables, saved model
