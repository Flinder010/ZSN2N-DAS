from obspy import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import h5py 

file = h5py.File( 'D:\Data\PoroTomo_iDAS16043_160321073721.h5' , 'r' ) 

data_file = file[ 'DasRawData' ]
dataset = data_file [ 'RawData' ]
st_raw = np.array(dataset)  #(30000,8720)   

st_raw_array_resize = st_raw[16000:17000,:1000]
matrix = st_raw_array_resize

plt.rcParams["figure.figsize"] = (9,9)
plt.title('Raw Data',fontsize=28)

plt.imshow(matrix, vmin=-1, vmax=1, cmap="seismic", aspect="auto", extent=[0,1000,1,0])
plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()