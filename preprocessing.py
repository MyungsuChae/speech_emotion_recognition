from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
import numpy as np
import pandas as pd
import openpyxl
import glob
import os
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use('ggplot')

wb = openpyxl.load_workbook("/data3/younghoon/Audio_data/voice_annotation_gender.xlsx")
ws = wb.get_sheet_by_name("Sheet1")

file_path = "/data3/mschae89/FlagshipAER/"
audio_path = "/data3/younghoon/Audio_data/mp4_file/"
Num_Frame = 1000    # max wave length (10 sec)
Stride = 0.01       # stride (10ms)
Window_size = 0.025 # filter window size (25ms)
Num_data = 10350    # Input data number
Num_mels = 40       # Mel filter number ############### if using squared, change line 22 & 43

Input_Raws=np.zeros((Num_data, Num_Frame, (Num_mels*2)), dtype = float)

for i in range(0, Num_data):
    audio_name = ws.cell(row=i+2, column=1).value
    y, sr = librosa.load(audio_path+audio_name+".mp4")
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=int(sr*Stride), n_fft=int(sr*Window_size), n_mels=Num_mels)

    S_squared = S * S
    S_conc = np.concatenate((S, S_squared), axis=0)
    r, c = S_conc.shape

    Input_Mels=np.zeros((r,Num_Frame), dtype=float)

    if c < Num_Frame :
        Input_Mels[:, :c] = S_conc[:, :c]
    else :
        Input_Mels[:, :Num_Frame] = S_conc[:, :Num_Frame]

    df = pd.DataFrame(Input_Mels.T)
    df.to_csv(file_path + "mels_40/" + audio_name + "_mels.csv", encoding='utf-8', index=False)

    Input_Raws[i, :Num_Frame, :r] = Input_Mels.T # (1000,40)


    if i%10==0:
        print(i)

np.save(file_path + "mels_40_0-99.npy", Input_Raws)
print('Pre-processing Done')