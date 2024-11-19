import obspy
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
from scipy.signal import butter, filtfilt
from scipy.signal import correlate
import scipy.ndimage
import scipy.signal
import matplotlib.patches as patches
from matplotlib.patches import Patch
from skimage.util import view_as_blocks, view_as_windows
import cv2

def correlate_func(x, idx, cc_thresh = 0.9):
    correlation = correlate(x[idx,:], x[(idx+1),:], mode="full")
    lags = np.arange(-(x[idx,:].size - 1), x[(idx+1),:].size)
    lag_idx = np.argmax(correlation)
    lag = lags[lag_idx]
    if lag > 0:
        if np.corrcoef(x[idx,lag:], x[(idx+1),:-lag], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([x[:(idx+1),:], np.zeros((x[:(idx+1),:].shape[0], lag))], axis=1),
                 np.concatenate([np.zeros((x[(idx+1):,:].shape[0], lag)), x[(idx+1):,:]], axis=1)],
                axis=0)
    if lag < 0:
        if np.corrcoef(x[idx,:-lag], x[(idx+1),lag:], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([np.zeros((x[:(idx+1),:].shape[0], abs(lag))), x[:(idx+1),:]], axis=1),
                 np.concatenate([x[(idx+1):,:], np.zeros((x[(idx+1):,:].shape[0], abs(lag)))], axis=1)],
                axis=0)

    return(x)

def correlate_func2(x, idx, cc_thresh = 0.9):
    correlation = correlate(x[:idx,:], x[(idx+1):(idx+2),:], mode="full", method="direct")
    idx_max_xcorr = np.argmax(np.amax(correlation, axis=1))

    lags = np.arange(-(x[idx_max_xcorr,:].size - 1), x[(idx+1),:].size)
    lag_idx = np.argmax(correlation[idx_max_xcorr,:])
    lag = lags[lag_idx]

    if lag > 0:
        if np.corrcoef(x[idx_max_xcorr,lag:], x[(idx+1),:-lag], rowvar=False)[0,1] > cc_thresh:
            idx_max_xcorr_start = np.amin(np.where(x[idx_max_xcorr,:] != 0))
            idx_plus1_start = np.amin(np.where(x[(idx+1),:] != 0))
            lag = lag - (idx_plus1_start - idx_max_xcorr_start)
            if lag > 0:
                x = np.concatenate(
                    [np.concatenate([x[:(idx+1),:], np.zeros((x[:(idx+1),:].shape[0], lag))], axis=1),
                     np.concatenate([np.zeros((x[(idx+1):,:].shape[0], lag)), x[(idx+1):,:]], axis=1)],
                    axis=0)
    if lag < 0:
        if np.corrcoef(x[idx,:-lag], x[(idx+1),lag:], rowvar=False)[0,1] > cc_thresh:
            idx_max_xcorr_start = np.amin(np.where(x[idx_max_xcorr,:] != 0))
            idx_plus1_start = np.amin(np.where(x[(idx+1),:] != 0))
            lag = lag + (idx_plus1_start - idx_max_xcorr_start)
            if lag < 0:
                x = np.concatenate(
                    [np.concatenate([np.zeros((x[:(idx+1),:].shape[0], abs(lag))), x[:(idx+1),:]], axis=1),
                     np.concatenate([x[(idx+1):,:], np.zeros((x[(idx+1):,:].shape[0], abs(lag)))], axis=1)],
                    axis=0)

    return(x)

def moving_window(data, window, func):
    wrapped = lambda region: func(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)

def marfurt_semblance(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces,nsamples = region.shape
    for i in range(ntraces-1):
        region = correlate_func(region, i, cc_thresh = 0.7)

    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_044207.903.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

denoised_matrix = cv2.blur(st_raw_array_resize, (5, 5))  # (5, 5) represents the size of the filter

no_chan = 13
no_samp = 19 

denoised_matrix_ex = moving_window(np.swapaxes(denoised_matrix, 0, 1), (no_chan, no_samp), marfurt_semblance)
denoised_matrix_ex = np.swapaxes(denoised_matrix_ex, 0, 1)

plt.rcParams["figure.figsize"] = (9,9) 

S = denoised_matrix_ex
SNR_loacl = S / (1.0 - S)
fig, ax = plt.subplots()
im = ax.imshow(SNR_loacl, vmin = 0, vmax = 15, cmap="magma_r", aspect="auto", extent=[0,S.shape[-1],S.shape[0]/1000,0]) 

rect1 = patches.Rectangle((100, 0.260), 550, 0.125, linewidth=4, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((100, 0.525), 550, 0.105, linewidth=4, edgecolor='m', facecolor='none')
rect3 = patches.Rectangle((100, 0.850), 550, 0.105, linewidth=4, edgecolor='b', facecolor='none')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.legend(handles=[Patch(facecolor='none', edgecolor='r', linewidth=2,label='Ten times of average SNR = ' + '{:.4f}'.format(10*np.mean(SNR_loacl[270:395, 100:650]))),
                   Patch(facecolor='none', edgecolor='m', linewidth=2,label='Ten times of average SNR = ' + '{:.4f}'.format(10*np.mean(SNR_loacl[435:640, 100:650]))),
                   Patch(facecolor='none', edgecolor='b', linewidth=2,label='Ten times of average SNR = ' + '{:.4f}'.format(10*np.mean(SNR_loacl[860:965, 100:650])))],
                    prop={'size': 20})

plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

fig.suptitle("(c)Mean Filtering(5×5)", fontsize=28) 
plt.show()