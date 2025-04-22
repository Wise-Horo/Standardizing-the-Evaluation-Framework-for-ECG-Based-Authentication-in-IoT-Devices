import math
import neurokit2 as nk
import pandas as pd
def extract_rpeaks_and_waves(ecg_signal,fs):
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="neurokit2", correct_artifacts=True)
    _, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")
    return rpeaks, waves

def extract_features(ecg_signal, rpeaks, waves):
    features = []
    R_list = rpeaks['ECG_R_Peaks']
    Q_list = waves['ECG_Q_Peaks']
    S_list = waves['ECG_S_Peaks']
    P_list = waves['ECG_P_Peaks']
    T_list = waves['ECG_T_Peaks']
    P_starts = waves['ECG_P_Onsets']
    T_ends = waves['ECG_T_Offsets']

    for i in range(len(R_list)):
        R, Q, S, P, T, PS, TE = R_list[i], Q_list[i], S_list[i], P_list[i], T_list[i], P_starts[i], T_ends[i]
        if not any(math.isnan(x) for x in [R, Q, S, P, T, PS, TE]):
            total_l = TE - PS
            AP = R - PS
            RQA = ecg_signal[R] - ecg_signal[Q]
            RSA = ecg_signal[R] - ecg_signal[S]
            features.append([total_l, AP, RQA, RSA])
    return features

def extract_AP(group):
    count = 0 
    AP = 0    
    for i in range(len(group)):
        if count < 30:
            AP += group["AP"].iloc[i]
            count+=1
        else:
            break
    return AP/count


def get_feature(group):
    AP = extract_AP(group)
    column_name = ['ID','RLP','RP','RQ','RS','RT','RTP','RQA','RSA','total_l']
    df = pd.DataFrame(columns = column_name)
    for i in range(len(group)):
        total_name = group["ID"].iloc[i]
        R = group['R'].iloc[i]
        PS = group['PS'].iloc[i]
        P = group['P'].iloc[i]-PS
        Q = group['Q'].iloc[i]-PS
        S = group['S'].iloc[i]-PS
        T = group['T'].iloc[i]-PS
        TE = group['TE'].iloc[i]-PS
        total_l = group['total_l'].iloc[i]
        RLP = (AP)/total_l
        RP = (AP-P)/total_l
        RQ = (AP-Q)/total_l
        RS = (S-AP)/total_l
        RT = (T-AP)/total_l
        RTP = (TE-AP)/total_l
        RQA = group['RQA'].iloc[i]
        RSA = group['RSA'].iloc[i]
        df.loc[len(df.index)] = [total_name,RLP,RP,RQ,RS,RT,RTP,RQA,RSA,total_l]
    return df

def authentication(ID, input_feature, stored_template):
    info = stored_template[ID]
    thresholds = [
        (info[0], 0.1),   # RLP_M
        (info[1], 0.188), # RP_M
        (info[2], 0.1709),# RQ_M
        (info[3], 0.129), # RS_M
        (info[4], 0.113)  # RT_M
    ]
    
    score = 0
    # 检查前5个指标，增加分数
    for (reference, tolerance), value in zip(thresholds, input_feature):
        upper = reference * (1 + tolerance)
        lower = reference * (1 - tolerance)
        if lower <= value <= upper:
            score += 1
        if score + len(thresholds) - (thresholds.index((reference, tolerance)) + 1) < 4:
            return False

    if score >= 4:
        RQA_M = info[5]
        RSA_M = info[6]
        if (RQA_M * 0.78 <= input_feature[5] <= RQA_M * 1.22) and (RSA_M * 0.83 <= input_feature[6] <= RSA_M * 1.17):
            return True

    return False