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

class network(nn.Module):
    def __init__(self,n_chan,chan_embed=48):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x

def pair_downsampler(img):
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(noisy_img):
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)

    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))

    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))

    loss = loss_res + loss_cons

    print(loss)

    return loss

def train(model, optimizer, noisy_img):

  loss = loss_func(noisy_img)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def denoise(model, noisy_img):

    with torch.no_grad():
            pred = torch.clamp( noisy_img - model(noisy_img), -40, 40 )
    return pred

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

def apply_border_division(matrix, border_size1, border_size2):      
    processed_matrix = matrix.copy()  
    processed_matrix[:border_size1, border_size2:-border_size2] /= 10 
    processed_matrix[-border_size1:, border_size2:-border_size2] /= 10  
    return processed_matrix  

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

example1 = "D:\Data\DAS-N2N-main\data\\BPT1_UTC_20200117_013019.232.mseed"
st_raw = read(example1)  

st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
for tr_no in range(len(st_raw)):
    st_raw_array[:,tr_no] = st_raw[tr_no].data

st_raw_array_resize = st_raw_array[:1000,:]
matrix = st_raw_array_resize

column_averages = np.mean(matrix, axis=0)
matrix_std = np.std(matrix, axis=0)
standard_matrix = (matrix - column_averages) / matrix_std

fs = 1000
lowcut = 30.0  
highcut = 100.0 
filtered_matrix = bandpass_filter(standard_matrix, lowcut, highcut, fs)
threshold = np.std(filtered_matrix) * 2.0
filtered_matrix[np.abs(filtered_matrix) < threshold] /= 5

border_size1 = 20 
border_size2 = 50
filtered_matrix = apply_border_division(filtered_matrix, border_size1, border_size2)

filtered_matrix = filtered_matrix.copy()
resize_image = torch.from_numpy(filtered_matrix).float()[None, None, :, :]  

n_chan = resize_image.shape[1]
model = network(n_chan)

max_epoch = 0
lr = 0.001        
step_size = 1000     
gamma = 0.5
i = 0          

model = torch.load("D:\PythonCode\Denoise\\Test_Generalized\\test1_model.pt")
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(max_epoch):
    train(model, optimizer, resize_image)
    scheduler.step()
    i+=1
    print(i)

denoised_img = denoise(model, resize_image)
#torch.save(model,'save_model.pt')

origin = resize_image.cpu().squeeze(0).permute(1,2,0)
denoised = denoised_img.cpu().squeeze(0).permute(1,2,0)
matrix_return = np.squeeze(denoised, axis=2) 
denoised_return = matrix_return * matrix_std

no_chan = 13
no_samp = 19 

S_ZSN2N_DAS_ex = moving_window(np.swapaxes(matrix_return, 0, 1), (no_chan, no_samp), marfurt_semblance)
S_ZSN2N_DAS_ex = np.swapaxes(S_ZSN2N_DAS_ex, 0, 1)

plt.rcParams["figure.figsize"] = (9,9) 

S = S_ZSN2N_DAS_ex
SNR_loacl = S / (1.0 - S)
fig, ax = plt.subplots()
im = ax.imshow(SNR_loacl, vmin = 0, vmax = 15, cmap="magma_r", aspect="auto", extent=[0,S.shape[-1],S.shape[0]/1000,0]) 

rect1 = patches.Rectangle((100, 0.305), 550, 0.095, linewidth=4, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((100, 0.450), 550, 0.095, linewidth=4, edgecolor='m', facecolor='none')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.legend(handles=[Patch(facecolor='none', edgecolor='r', linewidth=2,label='Max SNR in box = ' + '{:.4f}'.format(np.amax(SNR_loacl[305:400, 100:650]))),
                   Patch(facecolor='none', edgecolor='m', linewidth=2,label='Max SNR in box = ' + '{:.4f}'.format(np.amax(SNR_loacl[450:545, 100:650])))], 
                   prop={'size': 20})

plt.xlabel('DAS channel', fontsize=20)
plt.ylabel('Time(s)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

fig.suptitle("(f)ZSN2N-DAS", fontsize=28) 
plt.show()