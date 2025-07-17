import torch

# =======================
# Γενικές παραμέτρους
# =======================
in_channels = 9 * 3
d_model = 128
num_classes = 40
total_blocks = 6
seg_time = 50
batch_size = 8
max_epochs = 100
early_stopping_patience = 600
lr = 1e-4
dropout = 0.7

# =======================
# Δεδομένα
# =======================
subject_ids = range(1, 36)
channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
crop_len = 250
folder = "C:/path/to/your/benchmark_dataset"

# =======================
# Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
