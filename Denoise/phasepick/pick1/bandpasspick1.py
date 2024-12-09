import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy.signal import butter, filtfilt
import scipy

def butter_bandpass(lowcut, highcut, fs, order=5):  
    nyquist = 0.5 * fs  
    low = lowcut / nyquist  
    high = highcut / nyquist  
    b, a = butter(order, [low, high], btype='band')  
    return b, a 

def bandpass_filter(data, lowcut, highcut, fs, order=5):  
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  
    y = filtfilt(b, a, data, axis=0)  # setting axis = 0 represents filtering each column
    return y 

example1 = "D:\\Data\\DAS-N2N-main\\data\\BPT1_UTC_20200117_013019.232.mseed"
st_raw = read(example1)
 
st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))
for tr_no in range(len(st_raw)):
    st_raw_array[:, tr_no] = st_raw[tr_no].data
 
st_raw_array_resize = st_raw_array[:1000, :]
matrix = st_raw_array_resize
 
plt.rcParams["figure.figsize"] = (20, 6)  
 
channel_no = 365
arrival1 = 355/1000 # Arrival time in sec
arrival2 = 530/1000 # Arrival time in sec

fs = 1000
lowcut = 30.0  
highcut = 100.0 
filtered_matrix = bandpass_filter(matrix, lowcut, highcut, fs)

fig, ax = plt.subplots(1)  
 
ax.plot(np.arange(0, 1000) / 1000, filtered_matrix[:, channel_no], color='k', linewidth=2)
ax.axvline(x=arrival1, ls='--', color="red", linewidth=4)
ax.axvline(x=arrival2, ls='--', color="red", linewidth=4)
ylim = 25
ax.set_ylim(-ylim, ylim)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=20)
ax.set_ylabel("Strain rate", fontsize=20)
ax.set_xlabel('Time(s)', fontsize = 20)
 
ax.set_title("(d) 30-100Hz Bandpass", fontsize=24)  
 
plt.show()