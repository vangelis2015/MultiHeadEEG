import scipy.io
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, filtfilt
import scipy.stats as stats
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm

def riemannian_align_trials(trials):
    """
    Ευθυγραμμίζει EEG trials μέσω Riemannian Alignment.

    Args:
        trials: ndarray (n_trials, n_channels, n_times)

    Returns:
        aligned_trials: ndarray (n_trials, n_channels, n_times)
    """
    n_trials, C, T = trials.shape
    
    # 1. Υπολογισμός covariance matrices για όλα τα trials
    covs = np.array([trial @ trial.T / T for trial in trials])
    
    # 2. Riemannian mean
    C_bar = mean_riemann(covs)
    
    # 3. Υπολογισμός whitening matrix (inverse sqrt of mean)
    W = invsqrtm(C_bar)
    
    # 4. Εφαρμογή whitening στα trials (spatial filtering)
    aligned_trials = np.array([W @ trial for trial in trials])
    
    return aligned_trials



# def load_and_rearrange_mat_file(file_name):
#     """
#     Φορτώνει το αρχείο .mat, αναδιοργανώνει τη μεταβλητή 'data', αφαιρεί τις πρώτες και τελευταίες 125 χρονικές στιγμές,
#     και "κόβει" κάθε δείγμα σε 5 υπο-δείγματα των 250 χρονικών στιγμών.
    
#     Args:
#         file_name (str): Το όνομα του αρχείου .mat.
        
#     Returns:
#         tuple: 
#             - numpy.ndarray: Η αναδιοργανωμένη μεταβλητή 'data' στη μορφή [δείγματα, κανάλια, χρονικός χρόνος].
#             - list: Λίστα με τις ετικέτες των κλάσεων.
#     """
#     # Φόρτωση του αρχείου .mat
#     mat_data = scipy.io.loadmat(file_name)
    
#     # Έλεγχος αν η μεταβλητή 'data' υπάρχει στο αρχείο
#     if 'data' in mat_data:
#         data = mat_data['data']
#     else:
#         raise KeyError("Η μεταβλητή 'data' δεν βρέθηκε στο αρχείο.")
    
#     # Αφαίρεση των πρώτων και τελευταίων 125 χρονικών στιγμών (δηλαδή 126 έως 1375)
#     data = data[:, 125:1375, :, :]
    
#     # Αναδιοργάνωση των δεδομένων: [δείγματα (6 δείγματα ανά κλάση), κανάλια, χρόνος]
#     channels = data.shape[0]  # 64 κανάλια
#     time_points = data.shape[1]  # 1250 χρονικές στιγμές (μετά την αφαίρεση των 125 πρώτων και τελευταίων)
#     classes = data.shape[2]  # 40 κλάσεις
#     samples_per_class = data.shape[3]  # 6 δείγματα ανά κλάση
    
#     # "Κόψιμο" των δεδομένων σε 5 υπο-δείγματα των 250 χρονικών στιγμών
#     # Αντί να έχουμε 1250 χρονικές στιγμές, τώρα θα έχουμε 5 * 250 = 1250 χρονικές στιγμές για κάθε δείγμα
#     # Δημιουργία νέων δειγμάτων με 250 χρονικές στιγμές το καθένα
#     new_data = []
    
#     for i in range(classes):
#         for j in range(samples_per_class):
#             sample = data[:, :, i, j]  # Το αρχικό δείγμα (64 κανάλια x 1250 χρονικές στιγμές)
#             # Κόψιμο σε 5 υπο-δείγματα των 250 χρονικών στιγμών
#             for k in range(10):
#                 new_data.append(sample[:, k*125:(k+1)*125])  # Κόβουμε σε 5 μέρη των 250 χρονικών στιγμών

#     # Δημιουργία νέου πίνακα δεδομένων με τα κομμένα δείγματα
#     new_data = np.array(new_data)
    
#     # Δημιουργία ετικετών κλάσης (1 έως 40), κάθε κλάση εμφανίζεται 30 φορές (6 δείγματα ανά κλάση * 5 υπο-δείγματα)
#     new_labels = np.repeat(np.arange(1, classes + 1), samples_per_class * 10) - 1
    
#     return new_data, new_labels



# def generate_sine_and_cosine(frequencies, phases, sampling_rate, duration=1.0):
#     """
#     Δημιουργεί ημίτονα και συνημίτονα για δεδομένες συχνότητες και φάσεις.
    
#     Args:
#         frequencies (array-like): Οι συχνότητες για τις οποίες θα δημιουργήσουμε τα ημίτονα και συνημίτονα (σε Hz).
#         phases (array-like): Οι φάσεις για κάθε συχνότητα (σε rad).
#         sampling_rate (float): Η συχνότητα δειγματοληψίας (σε Hz).
#         duration (float): Η διάρκεια του σήματος (σε δευτερόλεπτα), προεπιλογή είναι 1.0 δευτερόλεπτο.
        
#     Returns:
#         tuple: Ένα tuple που περιέχει:
#             - numpy.ndarray: Ημίτονα για τις καθορισμένες συχνότητες και φάσεις.
#             - numpy.ndarray: Συνημίτονα για τις καθορισμένες συχνότητες και φάσεις.
#     """
#     # Δημιουργία του χρονικού άξονα
#     t = np.arange(0, duration, 1/sampling_rate)
    
#     # Αρχικοποιούμε τις λίστες για τα ημίτονα και συνημίτονα
#     sines = []
#     cosines = []
    
#     # Για κάθε συχνότητα και φάση υπολογίζουμε τα ημίτονα και συνημίτονα
#     for f, phase in zip(frequencies, phases):
#         # Δημιουργία του ημιτόνου και συνημίτονου για την κάθε συχνότητα και φάση
#         sine_wave = np.sin(2 * np.pi * f * t + phase)
#         cosine_wave = np.cos(2 * np.pi * f * t + phase)
        
#         # Προσθήκη στις αντίστοιχες λίστες
#         sines.append(sine_wave)
#         cosines.append(cosine_wave)
    
#     # Μετατροπή των λιστών σε πίνακες NumPy
#     sines = np.array(sines)
#     cosines = np.array(cosines)
    
#     return sines, cosines

def generate_sine_and_cosine(frequencies, phases, sampling_rate, duration=1.0, num_channels=1):
    """
    Δημιουργεί ημίτονα και συνημίτονα για δεδομένες συχνότητες, φάσεις και αριθμό καναλιών. Όλα τα κανάλια 
    έχουν τις ίδιες συχνότητες και φάσεις.
    
    Args:
        frequencies (numpy.ndarray): Οι συχνότητες για τις οποίες θα δημιουργήσουμε τα ημίτονα και συνημίτονα (σε Hz).
        phases (numpy.ndarray): Οι φάσεις για κάθε συχνότητα (σε rad).
        sampling_rate (float): Η συχνότητα δειγματοληψίας (σε Hz).
        duration (float): Η διάρκεια του σήματος (σε δευτερόλεπτα), προεπιλογή είναι 1.0 δευτερόλεπτο.
        num_channels (int): Ο αριθμός των καναλιών για τα οποία θέλουμε να δημιουργήσουμε τα σήματα.
        
    Returns:
        tuple: Ένα tuple που περιέχει:
            - numpy.ndarray: Ημίτονα για τις καθορισμένες συχνότητες και φάσεις για κάθε κανάλι.
            - numpy.ndarray: Συνημίτονα για τις καθορισμένες συχνότητες και φάσεις για κάθε κανάλι.
    """
    # Δημιουργία του χρονικού άξονα
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Δημιουργία ημιτόνων και συνημιτόνων για τις συχνότητες και φάσεις
    sine_wave = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * t + phases.reshape(-1, 1))  # (num_frequencies, num_time_points)
    cosine_wave = np.cos(2 * np.pi * frequencies.reshape(-1, 1) * t + phases.reshape(-1, 1))  # (num_frequencies, num_time_points)
    
    # Επαναλαμβάνουμε για τον αριθμό των καναλιών με βρόχο for
    sines = np.zeros((num_channels, sine_wave.shape[0], sine_wave.shape[1]))  # (num_channels, num_frequencies, num_time_points)
    cosines = np.zeros((num_channels, cosine_wave.shape[0], cosine_wave.shape[1]))  # (num_channels, num_frequencies, num_time_points)
    
    for ch in range(num_channels):
        sines[ch, :, :] = sine_wave  # Αντιγράφουμε το ίδιο σήμα για το κανάλι
        cosines[ch, :, :] = cosine_wave  # Αντιγράφουμε το ίδιο σήμα για το κανάλι
    
    return sines.transpose(1, 0, 2), cosines.transpose(1, 0, 2)
    

def normalize_sample(sample):
    """
    Κανονικοποιεί το δείγμα σε κάθε κανάλι ώστε να έχει μέση τιμή 0 και διακύμανση 1.
    
    Args:
        sample (numpy.ndarray): Το δείγμα προς κανονικοποίηση (κανάλια x χρονικές στιγμές).
        
    Returns:
        numpy.ndarray: Το κανονικοποιημένο δείγμα.
    """
    # Υπολογισμός μέσης τιμής και τυπικής απόκλισης για κάθε κανάλι
    mean = np.mean(sample, axis=1, keepdims=True)  # Μέση τιμή κατά μήκος των χρονικών στιγμών
    std = np.std(sample, axis=1, keepdims=True)  # Τυπική απόκλιση κατά μήκος των χρονικών στιγμών
    
    # Κανονικοποίηση: (x - mean) / std
    normalized_sample = (sample - mean) / std
    
    return normalized_sample

# def normalize_sample(sample):
#     """
#     Κανονικοποιεί το δείγμα σε κάθε κανάλι ώστε να έχει μέση τιμή 0 και εύρος [-1, 1].
    
#     Args:
#         sample (numpy.ndarray): Το δείγμα προς κανονικοποίηση (κανάλια x χρονικές στιγμές).
        
#     Returns:
#         numpy.ndarray: Το κανονικοποιημένο δείγμα.
#     """
#     # Υπολογισμός μέσης τιμής, ελάχιστης και μέγιστης τιμής για κάθε κανάλι
#     mean_val = np.mean(sample, axis=1, keepdims=True)  # Μέση τιμή κατά μήκος των χρονικών στιγμών
#     min_val = np.min(sample, axis=1, keepdims=True)  # Ελάχιστη τιμή
#     max_val = np.max(sample, axis=1, keepdims=True)  # Μέγιστη τιμή
    
#     # Κανονικοποίηση: (2 * (x - mean_val) / (max - min)) 
#     normalized_sample = 2 * (sample - min_val) / (max_val - min_val)-1
    
#     return normalized_sample



# def butter_bandpass(lowcut, highcut, fs, order=4):
#     """
#     Δημιουργία ενός band-pass φίλτρου Butterworth.
    
#     Args:
#         lowcut (float): Η χαμηλή συχνότητα της ζώνης (σε Hz).
#         highcut (float): Η υψηλή συχνότητα της ζώνης (σε Hz).
#         fs (float): Η συχνότητα δειγματοληψίας (σε Hz).
#         order (int): Η τάξη του φίλτρου.
        
#     Returns:
#         tuple: Συντελεστές του φίλτρου (b, a).
#     """
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

def apply_bandpass_filter(data, lowcut=6.0, highcut=45.0, fs=1000.0, order=10):
    """
    Εφαρμογή του band-pass φίλτρου στα δεδομένα.
    
    Args:
        data (numpy.ndarray): Τα δεδομένα προς φιλτράρισμα (κανάλια x χρονικές στιγμές).
        lowcut (float): Η χαμηλή συχνότητα της ζώνης (σε Hz).
        highcut (float): Η υψηλή συχνότητα της ζώνης (σε Hz).
        fs (float): Η συχνότητα δειγματοληψίας (σε Hz).
        order (int): Η τάξη του φίλτρου.
        
    Returns:
        numpy.ndarray: Τα φιλτραρισμένα δεδομένα.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    # Εφαρμογή του φίλτρου σε κάθε κανάλι
    #print('Fs=',fs)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

def load_and_rearrange_mat_file(file_name, lowcut=6.0, highcut=45.0, fs=1000.0):
    """
    Φορτώνει το αρχείο .mat, αναδιοργανώνει τη μεταβλητή 'data', αφαιρεί τις πρώτες και τελευταίες 125 χρονικές στιγμές,
    φιλτράρει τα δεδομένα στο εύρος συχνοτήτων 6-45 Hz, και "κόβει" κάθε δείγμα σε 5 υπο-δείγματα των 250 χρονικών στιγμών.
    
    Args:
        file_name (str): Το όνομα του αρχείου .mat.
        lowcut (float): Η χαμηλή συχνότητα της ζώνης (σε Hz).
        highcut (float): Η υψηλή συχνότητα της ζώνης (σε Hz).
        fs (float): Η συχνότητα δειγματοληψίας (σε Hz).
        
    Returns:
        tuple: 
            - numpy.ndarray: Η αναδιοργανωμένη και φιλτραρισμένη μεταβλητή 'data' στη μορφή [δείγματα, κανάλια, χρονικός χρόνος].
            - list: Λίστα με τις ετικέτες των κλάσεων.
    """
    # Φόρτωση του αρχείου .mat
    mat_data = scipy.io.loadmat(file_name)
    
    # Έλεγχος αν η μεταβλητή 'data' υπάρχει στο αρχείο
    if 'data' in mat_data:
        data = mat_data['data']
    else:
        raise KeyError("Η μεταβλητή 'data' δεν βρέθηκε στο αρχείο.")


    fr_file = 'C:/Users/vange/Documents/MatlabFiles/ssvep_benchmark_dataset/Freq_Phase.mat'
    fr_data = scipy.io.loadmat(fr_file)
    
    # Έλεγχος αν η μεταβλητή 'data' υπάρχει στο αρχείο
    if 'freqs' in fr_data and 'phases' in fr_data:
        freq_ssvep = fr_data['freqs']
        freq_ssvep = freq_ssvep[0]
        phases_ssvep = fr_data['phases']
        phases_ssvep = phases_ssvep[0]
    else:
        raise KeyError("Η μεταβλητή Freqs and Phases δεν βρέθηκε στο αρχείο.")

    
    # Αφαίρεση των πρώτων και τελευταίων 125 χρονικών στιγμών (δηλαδή 126 έως 1375)
    # matlab based chan_used = [48 54:58, 61:63] ;!!!!
    selected_channels = [47] + list(range(53, 58)) + list(range(60, 63))
    # selected_channels = list(range(60, 63))
    data = data[selected_channels, 125+34:1375+34, :, :]

    
    
    # Εφαρμογή του φίλτρου στις χρονικές στιγμές (και στα 64 κανάλια)
    channels = data.shape[0]  # 64 κανάλια
    time_points = data.shape[1]  # 1250 χρονικές στιγμές (μετά την αφαίρεση των 125 πρώτων και τελευταίων)
    classes = data.shape[2]  # 40 κλάσεις
    samples_per_class = data.shape[3]  # 6 δείγματα ανά κλάση
    bands = [(8,90),(24,90),(32,90)]#,(16,24),(24,32),(32,40)]#,(40,48),(48,56),(56,64),(64,72),(72,80),(80,88)]
    Nharmonic = 5#len(bands)

    # Δημιουργία ετικετών κλάσης (1 έως 40), κάθε κλάση εμφανίζεται 30 φορές (6 δείγματα ανά κλάση * 5 υπο-δείγματα)
    new_labels = np.repeat(np.arange(1, classes + 1), (samples_per_class) * 1)  - 1

    sines_all = []
    
    for nc in range(len(new_labels)):
        cls_idx = int(new_labels[nc])  # εξασφάλιση σωστού τύπου για indexing
        harmonics = []
        for i in range(Nharmonic):
            sines, cosines = generate_sine_and_cosine(
                (i + 1) * freq_ssvep[cls_idx],
                phases_ssvep[cls_idx],
                sampling_rate=fs,
                duration=0.5,
                num_channels=1
            )
            
            sines = sines.squeeze(1)       # shape: (1, T) → (T,)
            cosines = cosines.squeeze(1)   # shape: (1, T) → (T,)
            ref_sig = np.concatenate([sines, cosines], axis=0)  # shape: (2, T)
            harmonics.append(ref_sig)
        
        harmonics = np.concatenate(harmonics, axis=0)  # shape: (Nharmonic * 2, T)
        sines_all.append(harmonics)
    
    sines_all = np.stack(sines_all, axis=0)  # shape: (N_trials, Nharmonic*2, T)
    print('XXXX 1', sines_all.shape)

    # sines_all = np.concatenate(sines_all, axis=0)
    sines_all = np.stack(sines_all, axis=0)  # shape: (n_classes, Nharmonic*2, T)
    print(sines_all.shape)  # π.χ. (40, 6, 1000)
    
    print('XXXX 2',sines_all.shape)  # 👉 (18, 125)
        
    # sines_all = []
    # for i in range(Nharmonic):
    #     sines, cosines = generate_sine_and_cosine((i+1)*freq_ssvep, phases_ssvep, sampling_rate=fs, duration=0.5, num_channels=9)
    #     sines_all.append(sines)
    
    # sines_all = np.array(sines_all)
    # print(sines_all.shape)
    # D1,D2,D3,D4 = sines_all.shape
    # sines_all=sines_all.reshape(D2,D1*D3,D4)
    # print(sines_all.shape)
    # Φιλτράρισμα σε κάθε κανάλι (για κάθε δείγμα)
    filtered_data = []
    block_idx=[];
    for i in range(classes):
        for j in range(samples_per_class):
            
            sample = data[:, :, i, j]  # Το αρχικό δείγμα (64 κανάλια x 1250 χρονικές στιγμές)
            # filtered_sample = apply_bandpass_filter(sample, lowcut=lowcut, highcut=highcut, fs=fs)
            filtered_sample = apply_filter_bank(sample, fs, bands, order=10, return_index_map=False)
            # normalized_sample = normalize_sample(filtered_sample)
            
            # filtered_data.append(normalized_sample) 
            #Κόψιμο σε 10 υπο-δείγματα των 125 χρονικών στιγμών
            for k in range(1):
                
                # filtered_data.append(normalized_sample[:, k*125:(k+1)*125])  # Κόβουμε σε 5 μέρη των 250 χρονικών στιγμών
                sub_sample = filtered_sample[:, k*125:(k+1)*125]  # Κόβουμε σε 5 μέρη των 250 χρονικών στιγμών
                # Κανονικοποίηση του υπο-δειγματος
                normalized_sub_sample = normalize_sample(sub_sample)
                filtered_data.append(normalized_sub_sample)  # Προσθήκη του κανονικοποιημένου υπο-δειγματος
                block_idx.append(j) # block index
                
    
    # Δημιουργία νέου πίνακα δεδομένων με τα φιλτραρισμένα και κομμένα δείγματα
    filtered_data = np.array(filtered_data)
    block_idx = np.array(block_idx)
    

    
    
    return filtered_data, new_labels, sines_all, block_idx


def process_selected_mat_files(file_paths):
    """
    Επεξεργάζεται μόνο τα συγκεκριμένα αρχεία .mat που παρέχονται, και συνδυάζει τα δεδομένα και τις ετικέτες τους.
    
    Args:
        file_paths (list): Λίστα με τα μονοπάτια των επιλεγμένων αρχείων .mat.
        
    Returns:
        tuple: 
            - numpy.ndarray: Συνδυασμένα δεδομένα από τα επιλεγμένα αρχεία.
            - list: Συνδυασμένες ετικέτες από τα επιλεγμένα αρχεία.
    """
    all_data = []
    all_labels = []
    all_sines = []
    blk_idx = []
    for file_path in file_paths:
        # Φόρτωση και αναδιοργάνωση των δεδομένων από το αρχείο
        data, labels,ss,b_idx = load_and_rearrange_mat_file(file_path,fs=250)
        
        # Συγχώνευση των δεδομένων και των ετικετών
        all_data.append(data)
        all_labels.append(labels)
        all_sines.append(ss)
        blk_idx.append(b_idx)
    
    # Συνδυασμός όλων των δεδομένων σε έναν πίνακα και όλων των ετικετών σε μία λίστα
    all_data = np.vstack(all_data)  # Συγχώνευση των δεδομένων κάθε αρχείου
    all_sines = np.vstack(all_sines) 
    all_labels = np.concatenate(all_labels)  # Συγχώνευση των ετικετών κάθε αρχείου
    blk_idx = np.concatenate(blk_idx)
    
    return all_data, all_labels, all_sines, blk_idx


def filter_classes(data, labels, mSines, blockIdx, selected_classes):
    """
    Φιλτράρει τα δεδομένα για να κρατήσει μόνο τις επιλεγμένες κλάσεις.
    
    Args:
        data (numpy.ndarray): Τα δεδομένα.
        labels (numpy.ndarray): Οι ετικέτες των κλάσεων.
        selected_classes (list): Λίστα με τις κλάσεις που θέλεις να κρατήσεις.
        
    Returns:
        tuple: Φιλτραρισμένα δεδομένα και ετικέτες για τις επιλεγμένες κλάσεις.
    """
    # Δημιουργία μάσκας για τις επιλεγμένες κλάσεις
    mask = np.isin(labels, selected_classes)
    
    # Φιλτράρισμα των δεδομένων και των ετικετών
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    filtered_sines = mSines[mask]
    filtered_block = blockIdx[mask]
    
    return filtered_data, filtered_labels, filtered_sines, filtered_block


def butter_bandpass(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sos

def apply_filter_bank(trial, sfreq, bands, order=4, return_index_map=False):
    """
    Εφαρμόζει filter bank και επιστρέφει σήμα (B*C, T) + index_map (προαιρετικά).

    Args:
        trial (np.ndarray): EEG trial με shape (C, T)
        sfreq (float): Συχνότητα δειγματοληψίας
        bands (list of tuple): π.χ. [(6, 8), (8, 10), (10, 12)]
        order (int): Τάξη φίλτρου
        return_index_map (bool): Αν True, επιστρέφει και το index_map

    Returns:
        filtered: np.ndarray (B*C, T)
        index_map: list of tuples (band_idx, channel_idx) [αν ζητηθεί]
    """
    C, T = trial.shape
    B = len(bands)
    filtered = []
    index_map = []

    for b_idx, (low, high) in enumerate(bands):
        sos = butter_bandpass(low, high, fs=sfreq, order=order)
        filtered_band = sosfiltfilt(sos, trial, axis=1)  # (C, T)
        filtered.append(filtered_band)

        if return_index_map:
            for c_idx in range(C):
                index_map.append((b_idx, c_idx))

    filtered = np.stack(filtered, axis=0)    # (B, C, T)
    filtered = filtered.reshape(B * C, T)    # (B*C, T)

    if return_index_map:
        return filtered, index_map
    else:
        return filtered


def KL_div_channel(X_train,X_val,bins=30):
    C = X_train.shape[1]
    kl_scores= [];
    for c in range(C):
        t_train = X_train[:,c,:].flatten()
        t_val = X_val[:,c,:].flatten()

        hist_train, bins_edges = np.histogram(t_train,bins=bins, density=True)
        hist_val, _ = np.histogram(t_val,bins=bins, density=True)

        hist_train += 1e-10
        hist_val += 1e-10

        kl = stats.entropy(hist_train,hist_val)
        kl_scores.append(kl)

    return kl_scores


import os
import shutil

def reset_mlflow_local(mlruns_path="mlruns", confirm=True):
    """
    Deletes the local MLflow tracking directory (mlruns/) including all experiments and artifacts.

    Parameters:
        mlruns_path (str): Path to the MLflow tracking folder (default: 'mlruns').
        confirm (bool): Whether to ask for confirmation before deletion.

    Returns:
        None
    """
    if not os.path.exists(mlruns_path):
        print(f"📁 MLflow directory '{mlruns_path}' does not exist. Nothing to delete.")
        return

    if confirm:
        user_input = input(f"⚠️ Are you sure you want to DELETE the entire '{mlruns_path}/' folder? [yes/no]: ").strip().lower()
        if user_input != 'yes':
            print("❌ Deletion cancelled.")
            return

    shutil.rmtree(mlruns_path)
    print(f"✅ All MLflow data deleted from '{mlruns_path}/'. You now have a clean slate.")






        


