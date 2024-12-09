import numpy as np
from scipy.signal import find_peaks
from obspy import read
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_044207.903.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

denoised_matrix = cv2.GaussianBlur(st_raw_array_resize, (5, 5), 0)
matrix = denoised_matrix

sta_n = int(0.01 * matrix.shape[0])
lta_n = int(0.1 * matrix.shape[0])
threshold = 2
 
result = np.zeros_like(matrix, dtype=int)
 
for col in range(matrix.shape[1]):
    signal = matrix[:, col]
    sta = np.convolve(np.abs(signal), np.ones(sta_n) / sta_n, mode='same')
    lta = np.convolve(np.abs(signal), np.ones(lta_n) / lta_n, mode='same')
    lta[lta == 0] = np.finfo(float).eps
    sta_lta_ratio = sta / lta
    peaks, _ = find_peaks(sta_lta_ratio, height=threshold)
    result[peaks, col] = 1
 
rows_to_check1 = result[380:420, :]
rows_to_check2 = result[600:640, :]
rows_to_check3 = result[920:960, :]
count_of_arrivals1 = np.sum(rows_to_check1 == 1)
count_of_arrivals2 = np.sum(rows_to_check2 == 1)
count_of_arrivals3 = np.sum(rows_to_check3 == 1)

plt.rcParams["figure.figsize"] = (9,9)
fig, ax = plt.subplots()

im = ax.imshow(result, vmin=-1, vmax=1, cmap="seismic", aspect="auto", extent=[0,len(st_raw),1,0])

rect1 = patches.Rectangle((100, 0.260), 550, 0.125, linewidth=4, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((100, 0.525), 550, 0.105, linewidth=4, edgecolor='m', facecolor='none')
rect3 = patches.Rectangle((100, 0.850), 550, 0.105, linewidth=4, edgecolor='b', facecolor='none')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.legend(handles=[Patch(facecolor='none', edgecolor='r', linewidth=2,label='Arrivals in box = ' + format(count_of_arrivals1)),
                   Patch(facecolor='none', edgecolor='m', linewidth=2,label='Arrivals in box = ' + format(count_of_arrivals2)),
                   Patch(facecolor='none', edgecolor='b', linewidth=2,label='Arrivals in box = ' + format(count_of_arrivals3))], 
                   prop={'size': 20})

plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

fig.suptitle("(b)Gaussian Blur(5×5)", fontsize=28) 

plt.show()