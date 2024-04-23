import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import mne 
import pyedflib
from pyedflib import highlevel
from scipy.signal import resample
from pathlib import Path
import re

# Build Data files
data_path= r'D:/Graduate_School_Application/MS_Application_Projects/Neuro/TempleDataset/TUH_EEG_53gigs'
raw_path= data_path+r'/edf/train'
print(raw_path)

data_dict={}
edf_files=[]
for dirpath, _, filenames in os.walk(raw_path):
    for filename in filenames:
        if filename.endswith('tse'):
            annot_file=filename[:-3]+'tse'
            with open(dirpath+ '\\' + annot_file,'r') as file:
                lines=file.readlines()[2:] # events from 3rd line onwards
                lines = [line.rstrip('\n') for line in lines] # removing \n from each line
                sp1=dirpath.split('/')
                sp2=dirpath.split('\\') 
                for i,item in enumerate(lines):
                    annots=item.strip().split()
                    data_dict[f'{annot_file}_{i}']={
                        'Patient_id':sp2[3],'reference_type':sp2[1],'group':sp2[2],'session':sp2[4],'seizure_start': float(annots[0]),
                        'seizure_stop': float(annots[1]),'event_type': annots[2],'file_path': dirpath    
                    }
        elif filename.endswith('edf'):
            eeg_file=filename[:-3]+'edf'
            eeg_path=dirpath+'\\'+eeg_file
            edf_files.append(eeg_path)
                
data_dict

print(len(edf_files))

#Montages
montage_list= ['FP1-F7','F7-T3','T3-T5','T5-O1','FP2-F8','F8-T4','T4-T6','T6-O2','T3-C3','C3-CZ','CZ-C4','C4-T4','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2']
montage_list

valid_channels=[]
for montage in montage_list:
    x=montage.split(',')
    #print(x)
    for y in x:
        pair=y.split('-')
        anode=pair[0]
        valid_channels.append(anode)
        cathode=pair[1]
        valid_channels.append(cathode)
valid_channels


for edf_file in edf_files:
    eeg_data = pyedflib.EdfReader(edf_file)
    print(type(eeg_data))
    num_channels = eeg_data.signals_in_file
    print("NumberChannels",num_channels)
    channel_labels = eeg_data.getSignalLabels()
    print("ChannelLabel",channel_labels)
    sample_frequencies = eeg_data.getSampleFrequencies()  
    print("Frequency of the row:",sample_frequencies)
     
        
from scipy.signal import resample

def resample(signals, to_freq, window_size):
    new_samples = int(to_freq * window_size)
    resampled = resample(signals, new_samples, axis=1)
    return resampled

def get_signal_downsampled(channels,start,stop,freq_reqd):
    unique_channels= np.unique(valid_channels)
    samples = eeg_data.getNSamples()[0]
    signals= np.zeros(len(unique_channels),samples)
    for i,ch in enumerate(unique_channels):
        signals[i,:]=eeg_data.readSignal(ch)
            
    
    start, stop = float(start), float(stop)
    original_sample_frequency_row = eeg_data.getSampleFrequency(ch)
    original_start_index = int(np.floor(start * float(original_sample_frequency_row)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency_row)))

    seizure_signal = signals[original_start_index:original_stop_index]
        
    # Downsampling to 200 Hz
    new_lower_frequency = int(freq_reqd) # hardcoding for now.
    new_data_points = int(np.floor((stop - start) * new_lower_frequency))
    signal_downsampled = resample(seizure_signal, new_data_points)

    return signal_downsampled

def bipolar_signals_extract(montages,edf_start,edf_stop):
    bipolar_signals=[]
    for montage in montage_list:
        electrode_pair=montage.split(',')
        for electrodes in electrode_pair:
            electrode=electrodes.split('-')
            anode=electrode[0]
            freq_reqd=200
            downsampled_signal_from_anode= get_signal_downsampled(anode,start=edf_start,stop=edf_stop,freq_reqd=freq_reqd)
            cathode=electrode[1]
            downsampled_signal_from_cathode= get_signal_downsampled(cathode,start=edf_start,stop=edf_stop,freq_reqd=freq_reqd)
            difference= downsampled_signal_from_anode-downsampled_signal_from_cathode
            bipolar_signals.append(difference)
    bipolar_signals=np.array(bipolar_signals)
    
    return bipolar_signals        

    