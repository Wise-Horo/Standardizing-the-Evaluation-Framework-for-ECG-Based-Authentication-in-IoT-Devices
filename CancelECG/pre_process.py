import neurokit2 as nk
import numpy as np
from PIL import Image, ImageDraw
W_LEN = 256
W_LEN_1_4 = (256 // 4)+4
W_LEN_3_4 = 3 * (256 // 4)

def normalize_z_score(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def segment_heartbeats(ecg_signal, fs):
    # R-peak detection
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
    R_list = rpeaks['ECG_R_Peaks']
    entries = []
    # Extract individual heartbeat cycles around R-peaks
    for i in range(len(R_list)):
        r = R_list[i]
        if (r-W_LEN_1_4) < 0:
            continue
        if (r+W_LEN_3_4) > len(ecg_signal):
            break
        segmented_signal = list(ecg_signal[r-W_LEN_1_4:r+W_LEN_3_4])
        segmented_signal = list(normalize_z_score(segmented_signal))
        entries.append(segmented_signal)
    return entries

def heartbeat_to_image(heartbeat, image_size=224):
    heartbeat = heartbeat - np.min(heartbeat)
    if np.max(heartbeat) > 0:
        heartbeat = heartbeat / np.max(heartbeat)

    resampled_signal = np.interp(
        np.linspace(0, len(heartbeat)-1, image_size),
        np.arange(len(heartbeat)),
        heartbeat
    )

    img = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)

    y_coords = (1 - resampled_signal) * (image_size - 1)
    points = [(x, y) for x, y in enumerate(y_coords)]

    draw.line(points, fill=255, width=2)
    return img

def figure_generate(signal,fs):
    segments = segment_heartbeats(signal,fs)
    for i in range(len(segments)):
        heartbeat_signal = segments[i]
        img = heartbeat_to_image(heartbeat_signal)
        img.save(f"ecg_{i}.png")

