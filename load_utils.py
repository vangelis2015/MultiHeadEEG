import scipy.io
import numpy as np
import torch
# from scipy.signal import butter, filtfilt
import os

from scipy.signal import cheby1, butter, filtfilt

def bandpass_cheby1(data, fs, low, high, order=2, ripple=1):
    """
    Bandpass filtering με Chebyshev Type I IIR φίλτρο.
    
    Parameters:
        data   : np.ndarray
            Το σήμα εισόδου (τελευταίος άξονας = χρόνος)
        fs     : float
            Ρυθμός δειγματοληψίας (Hz)
        low    : float
            Κατώτερη συχνότητα αποκοπής (Hz)
        high   : float
            Ανώτερη συχνότητα αποκοπής (Hz)
        order  : int
            Τάξη του φίλτρου
        ripple : float
            Μέγιστη κυμάτωση στην passband (dB)

    Returns:
        np.ndarray: Το φιλτραρισμένο σήμα
    """
    nyq = fs / 2
    wn = [low / nyq, high / nyq]
    b, a = cheby1(order, ripple, wn, btype='band')
    return filtfilt(b, a, data, axis=-1)

def bandpass(data, fs, low, high, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_filterbank(signal, fs=250):
    # Εφαρμογή 3 υποζωνών όπως στο paper
    bands = [(8, 90), (16, 90), (24, 90)]
    filtered = []
    for low, high in bands:
        band = np.array([bandpass_cheby1(ch, fs, low, high) for ch in signal])
        filtered.append(band)
    return np.stack(filtered, axis=-1)  # shape: (channels, time, subbands)

def load_ssvep_mat(filepath, crop_len=256, channels=None):
    mat = scipy.io.loadmat(filepath)
    raw_data = mat['data']  # shape: (64, 1500, 40, 6)

    # mat = scipy.io.loadmat(filepath,struct_as_record=False, squeeze_me=True) #BETA dataset
    # raw_data = mat['data'].EEG  # shape: (64, 1500, 4, 40) #BETA dataset
    # raw_data = np.transpose(raw_data, (0,1,3,2))
    
    fs = 250

    n_channels, n_time, n_targets, n_trials = raw_data.shape
    n_time = 125 #len(125+34:1375+34), length = len(list(range(159, 1409, 1)))
    X, y = [], []

    for target in range(n_targets):
        for trial in range(n_trials):
            trial_data = raw_data[:, 125+34:125+2*125+34 , target, trial]  # shape: (64, time)
            if channels is not None:
                trial_data = trial_data[channels, :]

            # if trial_data.shape[1] >= crop_len:
            #     trial_data = trial_data[:, :crop_len]
            # else:
            #     pad_width = crop_len - trial_data.shape[1]
            #     trial_data = np.pad(trial_data, ((0, 0), (0, pad_width)), mode='edge')

            fb = apply_filterbank(trial_data, fs)  # (channels, time, 3)
            X.append(fb)
            y.append(target)

    X = np.array(X).astype(np.float32)  # (N, C, T, 3)
    y = np.array(y).astype(np.int64)
    # Debug info για σχήματα και labels
    # print("[✔] Loaded:", os.path.basename(filepath))
    # print("    → X shape:", X.shape)  # π.χ. (240, 9, 256, 3)
    # print("    → y shape:", y.shape)  # π.χ. (240,)
    # print("    → Unique labels:", np.unique(y))
    # print("    → Label counts:", {v: np.sum(y == v) for v in np.unique(y)})
    
    return torch.tensor(X), torch.tensor(y)

def load_all_subjects(folder, subject_ids, crop_len=250, channels=None):
    all_subjects = {}
    folder = os.path.abspath(folder)
    for sid in subject_ids:
        filename = f"S{sid:02d}.mat"
        filepath = os.path.join(folder, filename).replace("\\", "/")
        if os.path.exists(filepath):
            print(f"[✔] Loading: {filepath}")
            X, y = load_ssvep_mat(filepath, crop_len=crop_len, channels=channels)
            all_subjects[sid] = (X, y)
        else:
            print(f"[⚠] File not found: {filepath}")
    return all_subjects

def create_cross_subject_split(all_subjects, test_subject_id):
    X_train, y_train = [], []
    for sid, (X, y) in all_subjects.items():
        if sid == test_subject_id:
            X_test, y_test = X, y
        else:
            X_train.append(X)
            y_train.append(y)
    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)
    return X_train, y_train, X_test, y_test