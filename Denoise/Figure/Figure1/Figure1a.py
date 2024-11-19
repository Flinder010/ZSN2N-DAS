import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy.signal import butter, filtfilt

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_013019.232.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

plt.rcParams["figure.figsize"] = (9,9) 
plt.title('(a)Raw Date',fontsize=28)

plt.imshow(matrix, vmin=-40, vmax=40, cmap="seismic", aspect="auto", extent=[0,len(st_raw),1,0])
plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()