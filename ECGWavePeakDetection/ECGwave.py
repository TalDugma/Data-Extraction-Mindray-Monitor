import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

#start time
start = time.time()

# Load ECG data
ecg_df = pd.read_csv("Monitor/OR8___C400AD12-9394-3507-090C-1E0D010067C6/WaveformData/ECG_II-20230709124708~20230709145441.csv")
wave = ecg_df.iloc[:,2: ].values.ravel()
ecg_signal = wave
#save ecg signal
np.save("ecg_signal.npy",ecg_signal)

# Find R-peaks
_, rpeaks = nk.ecg_peaks(wave, sampling_rate=500)
R = rpeaks["ECG_R_Peaks"]
# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=500, method="peak")
#save the peaks
np.save("waves_peak.npy",waves_peak)
end = time.time()
print(f"Time taken: {end-start} seconds")
Q = waves_peak["ECG_Q_Peaks"]
S = waves_peak["ECG_S_Peaks"]
T = waves_peak["ECG_T_Peaks"]
P = waves_peak["ECG_P_Peaks"]

def example_plot(ecg_signal,R,Q,S,T,P):
    """
    PLOT FIRST 3 seconds OF ECG SIGNAL WITH R, Q, S, T, P PEAKS MARKED
    """
    plt.figure(figsize=(10,5))
    plt.plot(ecg_signal[:1500])
    plt.scatter(R[:3],ecg_signal[R[:3]],c='r',label='R peaks')
    plt.scatter(Q[:3],ecg_signal[Q[:3]],c='g',label='Q peaks')
    plt.scatter(S[:3],ecg_signal[S[:3]],c='b',label='S peaks')
    plt.scatter(T[:3],ecg_signal[T[:3]],c='y',label='T peaks')
    plt.scatter(P[:3],ecg_signal[P[:3]],c='m',label='P peaks')
    plt.legend()
    plt.savefig("ECG.png")

# example_plot(ecg_signal,R,Q,S,T,P)
