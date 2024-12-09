import matplotlib.pyplot as plt
import numpy as np
from obspy import read
import cv2
import torch
 
example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_044207.903.mseed"
st_raw = read(example1)
 
st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))
for tr_no in range(len(st_raw)):
    st_raw_array[:, tr_no] = st_raw[tr_no].data
 
st_raw_array_resize = st_raw_array[:1000, :]
matrix = st_raw_array_resize

denoised_matrix = cv2.GaussianBlur(matrix, (5, 5), 0) #(5, 5) represents the size of the filter and 0 indicates that the standard deviation should be automatically calculated based on the filter size.
tensor_matrix = torch.tensor(denoised_matrix)
 
plt.rcParams["figure.figsize"] = (20, 6)  
 
channel_no = 365
arrival1 = 356/1000
arrival2 = 552/1000
arrival3 = 813/1000
 
fig, ax = plt.subplots(1)  
 
ax.plot(np.arange(0, 1000) / 1000, tensor_matrix[:, channel_no], color='k', linewidth=2)
ax.axvline(x=arrival1, ls='--', color="red", linewidth=4)
ax.axvline(x=arrival2, ls='--', color="red", linewidth=4)
ax.axvline(x=arrival3, ls='--', color="red", linewidth=4)
ylim = torch.max(abs(tensor_matrix[:,channel_no])) * 1.05
ax.set_ylim(-ylim, ylim)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=20)
ax.set_ylabel("Strain rate", fontsize=20)
ax.set_xlabel('Time(s)', fontsize = 20)
 
ax.set_title("(b) Gaussian Blur(5×5)", fontsize=24)  
 
plt.show()