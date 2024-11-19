import obspy
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
from scipy.signal import butter, filtfilt
import h5py 
import cv2

file = h5py.File( 'D:\Data\Ridgecrest\\ci37447933.h5' , 'r' ) 
dataset = file[ 'data' ]
st_raw = np.array(dataset)  #(1150,12000) 
transposed_matrix = st_raw.T  

st_raw_array_resize = transposed_matrix[1:2001,:]
matrix = st_raw_array_resize
lowest_noise = np.min(matrix)
highest_noise = np.max(matrix)

filtered_matrix = cv2.GaussianBlur(matrix, (5, 5), 0)  

plt.rcParams["figure.figsize"] = (9,9)
plt.title('Gaussian Blur(5×5)',fontsize=28)

plt.imshow(filtered_matrix, vmin=-1, vmax=1, cmap="seismic", aspect="auto", extent=[0,len(st_raw),0,30])
plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()