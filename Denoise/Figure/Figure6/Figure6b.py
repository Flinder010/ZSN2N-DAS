import matplotlib.pyplot as plt
import numpy as np
from obspy import read
import scipy
import numpy as np
import torch
import cv2

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_044207.903.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

denoised_matrix = cv2.GaussianBlur(st_raw_array_resize, (5, 5), 0) #(5, 5) represents the size of the filter and 0 indicates that the standard deviation should be automatically calculated based on the filter size.
tensor_matrix = torch.tensor(denoised_matrix)

plt.rcParams["figure.figsize"] = (8,10) # Smaller fig size to make notebook size smaller

channel_no = 402
arrival1 = 345/1000
arrival2 = 533/1000
arrival3 = 798/1000

fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [1, 1]})
axs[0].plot(np.arange(0, 1000)/1000, tensor_matrix[:,channel_no], color='k', linewidth=2)
axs[0].axvline(x=arrival1, ls='--', color="red", linewidth=4)
axs[0].axvline(x=arrival2, ls='--', color="red", linewidth=4)
axs[0].axvline(x=arrival3, ls='--', color="red", linewidth=4)
ylim = torch.max(abs(tensor_matrix[:,channel_no])) * 1.05
axs[0].set_ylim(-ylim, ylim)
axs[0].set_xlim(0, 1)
axs[0].tick_params(labelsize=20)
axs[0].set_ylabel("Strain rate", fontsize=20)
axs[0].axes.xaxis.set_visible(False)

f, t, Sxx = scipy.signal.spectrogram(np.pad(tensor_matrix[:,channel_no], (50,50), "constant"), fs=1000, window = scipy.signal.windows.hamming(101), noverlap=100, nfft=1024, scaling='spectrum')
axs[1].pcolormesh(t-0.05, f, Sxx**(1/2), cmap="viridis")
axs[1].set_ylim(0, 200)
axs[1].set_ylabel('Frequency(Hz)', fontsize=20)
axs[1].set_xlabel('Time(s)', fontsize =20)
axs[1].tick_params(labelsize=20)

fig.suptitle("(b)Gaussian Blur(5×5)", fontsize=28) 
plt.subplots_adjust(wspace=0, hspace=0.05)

plt.show()