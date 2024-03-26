import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import sklearn as sk
#load ecg signal
ecg_signal = np.load("ecg_signal.npy")
#load waves peak
waves_peak = np.load("waves_peak.npy",allow_pickle=True).item()
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)
R = rpeaks["ECG_R_Peaks"]
Q = waves_peak["ECG_Q_Peaks"]
S = waves_peak["ECG_S_Peaks"]
T = waves_peak["ECG_T_Peaks"]
P = waves_peak["ECG_P_Peaks"]
# #plot R distances over time
# # print(R)
R_distances = np.diff(R)
# print(R_distances)
# plt.plot(R_distances[0:100])
# plt.title("R distances over time")
# plt.savefig("R_distances.png")
# # plot QRS distances over time
# plt.close()
QRS_distances = [s-q for s,q in zip(S,Q)]
# print()
# plt.plot(QRS_distances[0:100])
# plt.title("QRS distances over time")
# plt.savefig("QRS_distances.png")
# # plot R distances over time
# plt.plot(R_distances[0:100])
# plt.title("R distances over time")
# plt.savefig("R_distances.png")


plt.plot(ecg_signal) 
plt.savefig("full_ecg_signal.png")

