# -*- coding: utf-8 -*-
"""
This code is for seizure detection task on TUSZ dataset. Version is 1.5.2.
https://isip.piconepress.com/projects/tuh_eeg/ 

@author: Yuanda Zhu, PhD student at Georgia Institute of Technology, Atlanta, GA, USA
Year: 2023
"""

import os
import mne
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter, filtfilt, resample
from scipy.fft import rfft, rfftfreq

def get_all_TUH_EEG_session_paths(rootPath):
    session_paths = []
    all_patients = []
    prefixes = os.listdir(rootPath)
    for prefix in prefixes:
        patients = os.listdir(os.path.join(rootPath,prefix))
        for patient in patients:
            if patient not in all_patients:
                all_patients.append(patient)
            dates = os.listdir(os.path.join(rootPath,prefix,patient))
            for date in dates:
                files = os.listdir(os.path.join(rootPath,prefix,patient,date))
                sessions = []
                for file in files:
                    if file.endswith('.edf'):
                        sessions.append(file.split('.')[0])
                        
                for session in sessions:
                    session_paths.append(os.path.join(rootPath,prefix,patient,date,session+'.edf'))

    return session_paths, all_patients

def get_channels_from_raw(raw):
    montage_list_1 = ["EEG FP1-REF","EEG F7-REF","EEG T3-REF","EEG T5-REF",
                      "EEG FP2-REF","EEG F8-REF","EEG T4-REF","EEG T6-REF",
                      "EEG A1-REF","EEG T3-REF","EEG C3-REF","EEG CZ-REF",
                      "EEG C4-REF","EEG T4-REF","EEG FP1-REF","EEG F3-REF",
                      "EEG C3-REF","EEG P3-REF","EEG FP2-REF","EEG F4-REF",
                      "EEG C4-REF","EEG P4-REF"]
    
    montage_list_2 = ["EEG F7-REF","EEG T3-REF","EEG T5-REF","EEG O1-REF",
                      "EEG F8-REF","EEG T4-REF","EEG T6-REF","EEG O2-REF",
                      "EEG T3-REF","EEG C3-REF","EEG CZ-REF","EEG C4-REF",
                      "EEG T4-REF","EEG A2-REF","EEG F3-REF","EEG C3-REF",
                      "EEG P3-REF","EEG O1-REF","EEG F4-REF","EEG C4-REF",
                      "EEG P4-REF","EEG O2-REF"]
    
    montage_indices_1 = [raw.ch_names.index(ch) for ch in montage_list_1]
    montage_indices_2 = [raw.ch_names.index(ch) for ch in montage_list_2]

    try:    
        signals_1 = raw.get_data(picks=montage_indices_1)
        signals_2 = raw.get_data(picks=montage_indices_2)
    except:
        print('Something is wrong when reading channels of the raw EEG signal')
        flag_wrong = True
        return flag_wrong, 0
    else:
        flag_wrong = False
    
    return flag_wrong, signals_1-signals_2
    
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def slice_signals_into_binary_segments(signals, thisFS, labels, segment_interval, seizure_types, seizure_overlapping_ratio):
    
    # This "segments" variable is to store seizure segments according to each seizure class
    segments = [[] for i in range(len(seizure_types))]

    for this_label in labels:
        if this_label[2] == 'bckg':
            label_index = 0
        else:
            label_index = 1
        
        # This "seg" variable is to all segments of one line of labels
        seg = []
        for i in range(this_label[0]*thisFS, this_label[1]*thisFS, int(segment_interval*(1-seizure_overlapping_ratio[label_index])*thisFS)):
            
            if i+segment_interval*thisFS > this_label[1]*thisFS:
                break
            
            one_window = []
            noise_flag = False
            incomplete_flag = False
            for j in range(len(signals)):
                this_channel = signals[j][i:i+segment_interval*thisFS]
                # print(len(this_channel))
                if len(this_channel) < segment_interval*thisFS:
                    incomplete_flag = True
                    break
                # print(max(abs(this_channel)))
                if max(abs(this_channel)) > 500/10**6:
                    noise_flag = True
                    break
                one_window.append(this_channel)
                
            # seg.append(np.array(one_window))
            if incomplete_flag==False and noise_flag==False and one_window and len(one_window[0]) == thisFS*segment_interval:
                seg.append(np.array(one_window))
        
        # if this_label[2] in seizure_types:            
        #     this_index = seizure_types.index(this_label[2])
        segments[label_index].append(seg)
    
    return segments  

def slice_signals_into_multiclass_segments(signals, thisFS, labels, segment_interval, seizure_types, seizure_overlapping_ratio):
    
    # This "segments" variable is to store seizure segments according to each seizure class
    segments = [[] for i in range(len(seizure_types))]

    for this_label in labels:
        if this_label[2] not in seizure_types:
            print('Seizure type not included: ', this_label[2])
            continue
        label_index = seizure_types.index(this_label[2])
        
        # This "seg" variable is to all segments of one line of labels
        seg = []
        for i in range(this_label[0]*thisFS, this_label[1]*thisFS, int(segment_interval*(1-seizure_overlapping_ratio[label_index])*thisFS)):
            
            if i+segment_interval*thisFS > this_label[1]*thisFS:
                break
            
            one_window = []
            noise_flag = False
            incomplete_flag = False
            for j in range(len(signals)):
                this_channel = signals[j][i:i+segment_interval*thisFS]
                # print(len(this_channel))
                if len(this_channel) < segment_interval*thisFS:
                    incomplete_flag = True
                    break
                # print(max(abs(this_channel)))
                if max(abs(this_channel)) > 500/10**6:
                    noise_flag = True
                    break
                one_window.append(this_channel)
                
            # seg.append(np.array(one_window))
            if incomplete_flag==False and noise_flag==False and one_window and len(one_window[0]) == thisFS*segment_interval:
                seg.append(np.array(one_window))
        
        # if this_label[2] in seizure_types:            
        #     this_index = seizure_types.index(this_label[2])
        segments[label_index].append(seg)
    
    return segments  

def resample_data_in_each_channel(signals, thisFS, resampleFS):
    
    sigResampled = []
    # num = int(len(signals[0])/thisFS*resampleFS)
    for sig in signals:
        if type(sig) == np.ndarray:
            num = int(sig.shape[0]/thisFS*resampleFS)
        else:
            num = int(len(sig)/thisFS*resampleFS)
        y = resample(sig, num)
        sigResampled.append(y)
    
    return sigResampled

def plot_signal_in_frequency(signal, filtered_signal, sample_rate):
    # Suppose `signal` is your signal data, `filtered_signal` is the filtered data
    # signal = ...
    # filtered_signal = ...

    # Compute the frequency representation of the signals
    fft_orig = rfft(signal)
    fft_filtered = rfft(filtered_signal)

    # Compute the frequencies corresponding to the FFT output elements
    freqs = rfftfreq(len(signal), 1/sample_rate)

    # Plot the original signal in frequency domain
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, np.abs(fft_orig))
    plt.title('Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Plot the filtered signal in frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(fft_filtered))
    plt.title('Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def make_a_filtered_plot_for_comparison(signals, filtered_signals, thisFS):
    plt.figure()
    plt.clf()
    maximum_samples = 200
    channel_index = 5
    if maximum_samples == -1:
        t = np.linspace(0, signals.shape[1]/thisFS, signals.shape[1])
        plt.plot(t, signals[channel_index,:], label='Noisy signal')
        plt.plot(t, filtered_signals[channel_index][:], label='Filtered signal')
    else: 
        t = np.linspace(0, maximum_samples/thisFS, maximum_samples)    
        plt.plot(t, signals[channel_index,:maximum_samples], label='Noisy signal')
        plt.plot(t, filtered_signals[channel_index][:maximum_samples], label='Filtered signal')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    # plt.show()
    plt.savefig('filtered_signal_plot.png')
    
    return

def make_a_resampled_plot_for_comparison(filtered_signals, resampled_signals, thisFS, resampleFS):
    plt.figure()
    plt.clf()
    maximum_duration = 1 # unit: second
    channel_index = 5
    if maximum_duration == -1:
        t = np.linspace(0, filtered_signals.shape[1]/thisFS, filtered_signals.shape[1])
        plt.plot(t, filtered_signals[channel_index][:], label='Filtered signal')
        plt.plot(t, resampled_signals[channel_index][:], label='Resampled signal')
    else: 
        tf = np.linspace(0, maximum_duration, maximum_duration*thisFS)
        plt.plot(tf, filtered_signals[channel_index][:maximum_duration*thisFS], label='Filtered signal')
        tr = np.linspace(0, maximum_duration, maximum_duration*resampleFS)
        plt.plot(tr, resampled_signals[channel_index][:maximum_duration*resampleFS], label='Resampled signal')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    # plt.show()
    plt.savefig('resampled_signal_plot.png')
    
    return