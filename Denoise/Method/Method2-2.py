from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import butter, filtfilt

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

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_013019.232.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

fs = 1000
lowcut = 30.0  
highcut = 100.0 
filtered_matrix = bandpass_filter(matrix, lowcut, highcut, fs)

stride = 2  

kernel = np.array([[0, 0.5],  [0.5, 0]])  

output_matrix = np.zeros((500, 493)) 
   
for i in range(0, filtered_matrix.shape[0] - kernel.shape[0] + 1, stride):  
    for j in range(0, filtered_matrix.shape[1] - kernel.shape[1] + 1, stride):  
        region = filtered_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]]    
        output_matrix[i//stride, j//stride] = np.sum(region * kernel)  

plt.rcParams["figure.figsize"] = (9,9)
plt.title('Noise2',fontsize=28)

plt.imshow(output_matrix, vmin=-40, vmax=40, cmap="seismic", aspect="auto", extent=[0,493,0.5,0])
plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()