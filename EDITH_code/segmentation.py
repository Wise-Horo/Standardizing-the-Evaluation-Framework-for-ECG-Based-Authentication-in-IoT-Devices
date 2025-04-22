import numpy as np
import neurokit2 as nk
W_LEN = 256
W_LEN_1_4 = 256 // 4
W_LEN_3_4 = 3 * (256 // 4)
def normalize_z_score(signal):
    return (signal - np.mean(signal)) / np.std(signal)
def segmentation(ecg_signal,r_peaks_list):
    R_list = r_peaks_list
    for i in range(len(R_list)):
        r = R_list[i]
        if (r-W_LEN_1_4) < 0:
            continue
        if (r+W_LEN_3_4) > len(ecg_signal):
            break
    segmented_signal = list(ecg_signal[r-W_LEN_1_4:r+W_LEN_3_4])
    segmented_signal = list(normalize_z_score(segmented_signal))
    return segmented_signal